###################(Opus)#########################
img_feats = self.extract_feat(img, img_metas)
outs = self.pts_bbox_head(img_feats, img_metas)
loss_inputs = [voxel_semantics, mask_camera, outs]
losses = self.pts_bbox_head.loss(*loss_inputs)
############################################
"""OPUS_HEAD"""
B, Q, = mlvl_feats[0].shape[0], self.num_query
init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
query_feat = init_points.new_zeros(B, Q, self.embed_dims)

#############################################
cls_scores, refine_pts = self.transformer(
    init_points,
    query_feat,
    mlvl_feats,
    img_metas=img_metas,
)
#############################################
"transformer"
cls_scores, refine_pts = self.decoder(
    query_points, query_feat, mlvl_feats, img_metas)
cls_scores = [torch.nan_to_num(score) for score in cls_scores]
refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]
return cls_scores, refine_pt
#############################################

"""def forward(self, query_points, query_feat, mlvl_feats, img_metas):"""
cls_scores, refine_pts = [], []
# organize projections matrix and copy to CUDA
lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
lidar2img = query_feat.new_tensor(lidar2img) # [B, N, 4, 4]
ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
ego2lidar = query_feat.new_tensor(ego2lidar) # [B, 4, 4]
ego2lidar = ego2lidar.unsqueeze(1).expand_as(lidar2img)  # [B, N, 4, 4]
occ2img = torch.matmul(lidar2img, ego2lidar)
# group image features in advance for sampling, see `sampling_4d` for more details
for lvl, feat in enumerate(mlvl_feats):
    B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
    N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC//self.num_groups
    assert T*N == TN
    feat = feat.reshape(B, T, N, G, C, H, W)
    if MSMV_CUDA:  # Our CUDA operator requires channel_last
        feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
        feat = feat.reshape(B*T*G, N, H, W, C)
    else:  # Torch's grid_sample requires channel_first
        feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
        feat = feat.reshape(B*T*G, C, N, H, W)
    mlvl_feats[lvl] = feat.contiguous()
for i, decoder_layer in enumerate(self.decoder_layers):
    DUMP.stage_count = i
    query_points = query_points.detach()
    query_feat, cls_score, query_points = decoder_layer(
        query_points, query_feat, mlvl_feats, occ2img, img_metas)
    cls_scores.append(cls_score)
    refine_pts.append(query_points)
return cls_scores, refine_pts
#############################################

#############################################

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        """
        query_points: [B, Q, 3] [x, y, z]
        """
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)  # [B, Q, P * num_classes]
        reg_offset = self.scale * self.reg_branch(query_feat)  # [B, Q, P * 3]
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)

        if DUMP.enabled:
            pass # TODO: enable OTR dump

        return query_feat, cls_score, refine_pt
        
        #############################################
        
##自注意力计算过程 self.self_attn(query_points, query_feat)
        dist = self.calc_points_dists(query_points)# 计算每个query之间的距离  用query对应点的均值代表query的位置
        tau = self.gen_tau(query_feat)  # [B, Q, 8]  # query经过Linear层，预测每个头 attn_mask的权重

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]  # 两者点乘，得到每个头自注意力的 mask矩阵

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask) #使用transformer的自注意力模块
    
##采样的过程sampled_feat = self.sampling(query_points, query_feat, mlvl_feats, occ2img, img_metas)
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range) # 获得每个query预测点集的均值和方差
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        # bs num_query 48
        sampling_offset = self.sampling_offset(query_feat)  # 每个query 经过nn.Linear(embed_dims, num_groups * num_points * 3)预测偏移
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        sampling_points = query_center + sampling_offset * query_scale
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h, image_w,
            self.num_views
        )  # [B, Q, G, FP, C]
