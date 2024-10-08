import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from .bbox.utils import decode_points
from scipy.spatial import cKDTree

@HEADS.register_module()
class OPUSHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10, 
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()
    
    def get_dn(self,voxel_semantics,mask_camera,num_query,points_per_query):
        gt_points_list, gt_masks_list, gt_labels_list = self.get_sparse_voxels(voxel_semantics=voxel_semantics,mask_camera=mask_camera)
        bs = len(gt_points_list)
        assert len(gt_points_list)==len(gt_masks_list)==len(gt_labels_list)
        
        dngt_points = []
        dngt_label = []
        asis_points = []
        for i in range(bs):
            gt_points = gt_points_list[i][gt_masks_list[i]]
            gt_labels = gt_labels_list[i][gt_masks_list[i]]
            # 生成每个query对应的初始点 当前策略：随机挑选
            selec_indice = torch.randperm(gt_points.shape[0])[:num_query]
            dn_points = gt_points[selec_indice]
            
            #构建树进行搜索
            tree = cKDTree(gt_points.cpu())
            
            nearest_neighbors = gt_points_list[0].new_zeros(num_query,points_per_query,3)
            dn_points_label =  gt_labels_list[0].new_zeros(num_query,points_per_query)
                # 对每个query执行 KNN 搜索，寻找128个最近邻点 包括自己  
            for i, point in enumerate(dn_points):  
                # 查询最近的points_per_query个点（包括自己），然后去掉自己（如果需要）  
                _, indices = tree.query(point.cpu(), k=points_per_query)  
                # # 移除可能是自己的第一个索引（如果 clouds 中包含 input 的点）  
                # indices = indices[1:]  
                # 如果不足 128 个点，则只取可用的  
                nearest_neighbors[i, :len(indices), :] = gt_points[indices]  
                dn_points_label[i,:len(indices)] = gt_labels[indices]
            asis_points.append(dn_points[None,...])
            dngt_points.append(nearest_neighbors[None,...])
            dngt_label.append(dn_points_label[None,...])
        out = {'dn_points':torch.cat(asis_points,dim=0),
               'dngt_points':torch.cat(dngt_points,dim=0),
               'dngt_label':torch.cat(dngt_label,dim=0)
               }
        return out
        
    def forward(self, mlvl_feats, img_metas,voxel_semantics,mask_camera):
        B, Q, = mlvl_feats[0].shape[0], self.num_query # mlvl_feats四种分辨率的采样特征 bs 48 256 H W。48是什么？
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1) #bs query num_points 3
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)
        ############################################################################################################################
        num_query = 100
        points_per_query = 128
        dn_meta = self.get_dn(voxel_semantics,mask_camera,num_query,points_per_query)
        
        dn_points = dn_meta['dn_points'] #此时已经为真是尺度
        dn_points = dn_points[:,:,None,:]
        dngt_points = dn_meta['dngt_points']
        dngt_label = dn_meta['dngt_label']
        dn_query = dn_points.new_zeros(dn_points.shape[0],dn_points.shape[1],self.embed_dims)
        ############################################################################################################################
        cls_scores, refine_pts, dn_cls_scores, dn_refine_pts = self.transformer(
            init_points,
            query_feat,
            dn_points,
            dn_query,
            mlvl_feats,
            img_metas=img_metas,
        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    dn_all_cls_scores=dn_cls_scores,
                    dn_all_refine_pts=dn_refine_pts,
                    dngt_points = dngt_points,
                    dngt_label = dngt_label)

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1
    
    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)
        label_weights = cls_weights * \
            self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx) & gt_masks
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)

        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights, 
                gt_pts_weights)
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0) # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
             self._get_target_single, refine_pts_list, gt_points_list, 
             gt_masks_list, gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']
        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        
        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list)
        ################################################################
        dn_all_cls_scores = preds_dicts['dn_all_cls_scores'] 
        dn_all_refine_pts = preds_dicts['dn_all_refine_pts']
        dngt_points = preds_dicts['dngt_points']
        dngt_label = preds_dicts['dngt_label']
        dngt_points_list = []
        dngt_label_list = []
        dngt_mask_list = []
        for i in range(dngt_points.shape[0]):
            dngt_points_list.append(dngt_points[i].flatten(0,1))
            dngt_label_list.append(dngt_label[i].flatten(0,1))
            dngt_mask_list.append(torch.full_like(dngt_label[i].flatten(0,1),True,dtype=bool))
            
        all_dngt_points_list = [dngt_points_list for _ in range(num_dec_layers)]
        all_dngt_label_list = [dngt_label_list for _ in range(num_dec_layers)]
        all_dngt_mask_list = [dngt_mask_list for _ in range(num_dec_layers)]
        
        
        dn_losses_cls, dn_losses_pts = multi_apply(
            self.loss_single, dn_all_cls_scores, dn_all_refine_pts, 
            all_dngt_points_list, all_dngt_mask_list, all_dngt_label_list)
        #################################################################################
        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, 
                gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
#####################################################
        loss_dict["dn_loss_cls"]=dn_losses_cls[-1]
        loss_dict["dn_loss_pts"]=dn_losses_pts[-1]
        
#####################################################
        # loss from other decoder layers  
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        num_dn_dec_layer=0
        for dn_loss_cls_i, dn_loss_pts_i in zip(dn_losses_cls[:-1], dn_losses_pts[:-1]):
            loss_dict[f'd{num_dn_dec_layer}.dn_loss_cls'] = dn_loss_cls_i
            loss_dict[f'd{num_dn_dec_layer}.dn_loss_pts'] = dn_loss_pts_i
            num_dn_dec_layer += 1
        return loss_dict
    
    def get_occ(self, pred_dicts, img_metas, rescale=False):
        all_cls_scores = pred_dicts['all_cls_scores']
        all_refine_pts = pred_dicts['all_refine_pts']
        cls_scores = all_cls_scores[-1].sigmoid()
        refine_pts = all_refine_pts[-1]

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)
        score_thr = self.test_cfg.get('score_thr', 0.)

        result_list = []
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = decode_points(refine_pts, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            mask_score = (cls_scores > score_thr).any(dim=-1)
            mask = mask_dist & mask_score
            refine_pts = refine_pts[mask]
            cls_scores = cls_scores[mask]

            pts = torch.cat([refine_pts, cls_scores], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1]).long()
            pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
            scores = scores.sum(dim=1) / num_pts[..., None]

            if self.test_cfg.get('padding', True):
                occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1], 
                                        self.voxel_num[2], self.num_classes))
                occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
                occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
                # repalce with original occ prediction
                original_mask = (occ > score_thr).any(dim=1, keepdim=True)
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
                # sparse dense occ
                eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
                voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
                scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]

            labels = scores.argmax(dim=-1)
            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=voxels.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_masks, gt_labels
