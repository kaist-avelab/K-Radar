import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
import numpy as np

class MeanVoxelEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.roi = self.cfg.DATASET.LPC.ROI
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']
        
        # additional mlps for NUM_FILTERS e.g., [64] to [32, 64]
        self.num_point_features = cfg.MODEL.VOXEL_ENCODER.NUM_POINT_FEATURES
        self.num_filters = cfg.MODEL.VOXEL_ENCODER.NUM_FILTERS
        
        # simplified pointnet in self.proj in pointpillar_backbone
        # io_channels = [self.num_point_features] + self.num_filters
        # self.voxel_encoder = nn.ModuleList()
        # for i in range(len(io_channels)-1):
        #     self.voxel_encoder.append(nn.Sequential(
        #         nn.Linear(io_channels[i], io_channels[i+1]),
        #         nn.ReLU()
        #     ))

        # PointToVoxel
        self.pc_range = [x_min, y_min, z_min, x_max, y_max, z_max]
        self.voxel_size = self.cfg.MODEL.VOXEL_ENCODER.VOXEL_SIZE
        self.max_num_voxels = self.cfg.MODEL.VOXEL_ENCODER.MAX_NUM_VOXELS
        self.max_num_pts_per_voxels = self.cfg.MODEL.VOXEL_ENCODER.MAX_NUM_PTS_PER_VOXELS
        self.gen_voxels = PointToVoxel(
            vsize_xyz = self.voxel_size,
            coors_range_xyz = self.pc_range,
            num_point_features = self.num_point_features,
            max_num_voxels = self.max_num_voxels,
            max_num_points_per_voxel = self.max_num_pts_per_voxels,
            device = torch.device('cuda') # Assuming single GPU
        )

    def forward(self, data_dic):
        """
        Args:
            data_dic:
                voxels: num_voxels x max_points_per_voxel x C_points
                voxel_num_points: optional (num_voxels)
        Returns:
            vfe_features: (num_voxels, C)
        """
        ldr_pc_64 = data_dic['ldr_pc_64'].cuda()
        pts_batch_indices = data_dic['pts_batch_indices_ldr_pc_64'].cuda()

        batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []

        for batch_id in range(data_dic['batch_size']):
            pc = ldr_pc_64[torch.where(pts_batch_indices == batch_id)].view(-1, 4)
            voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(pc)
            voxel_batch_id = torch.full((voxel_coords.shape[0], 1), batch_id, device=ldr_pc_64.device, dtype=torch.int64)
            voxel_coords = torch.cat((voxel_batch_id, voxel_coords), dim=-1)
            
            batch_voxel_features.append(voxel_features)
            batch_voxel_coords.append(voxel_coords)
            batch_num_pts_in_voxels.append(voxel_num_points)

        voxel_features, voxel_coords, voxel_num_points = torch.cat(batch_voxel_features), torch.cat(batch_voxel_coords), torch.cat(batch_num_pts_in_voxels)
        data_dic['voxel_features'], data_dic['voxel_coords'], data_dic['voxel_num_points'] = voxel_features, voxel_coords, voxel_num_points
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        encoded_voxels = points_mean / normalizer # BM x C

        # Simplified pointnet
        # for encoder in self.voxel_encoder:
        #     encoded_voxels = encoder(encoded_voxels)

        data_dic['encoded_voxel_features'] = encoded_voxels.contiguous()

        ### Additional Points Preprocessing for PVRCN_PP ###
        # pts = data_dic['ldr_pc_64']
        # pts_indices = data_dic['pts_batch_indices_ldr_pc_64']
        # pts_coords = torch.cat((pts_indices.unsqueeze(1), pts), dim = -1)
        # data_dic['point_coords'] = pts_coords[:, :4].cuda()
        # data_dic['points'] = pts_coords.cuda() # N x (batch_ind, x, y, z, C)
        ### Additional Points Preprocessing for PVRCN_PP ###

        return data_dic
