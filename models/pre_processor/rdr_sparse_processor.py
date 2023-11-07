'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn


from spconv.pytorch.utils import PointToVoxel

class RadarSparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarSparseProcessor, self).__init__()
        self.cfg = cfg

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        if self.cfg_dataset_ver2:
            cfg_ds = self.cfg.DATASET
            roi = cfg_ds.roi
            x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
            self.min_roi = [x_min, y_min, z_min]
            self.grid_size = roi.grid_size
            self.input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

            self.is_with_simplified_pointnet = cfg.MODEL.PRE_PROCESSOR.SIMPLIFIED_POINTNET.IS_WITH_SIMPLIFIED_POINTNET
            if self.is_with_simplified_pointnet:
                out_channel = cfg.MODEL.PRE_PROCESSOR.SIMPLIFIED_POINTNET.OUT_CHANNEL
                cfg.MODEL.PRE_PROCESSOR.INPUT_DIM = out_channel
                self.simplified_pointnet = nn.Linear(self.input_dim, out_channel, bias=False)
                self.pooling_method = cfg.MODEL.PRE_PROCESSOR.SIMPLIFIED_POINTNET.POOLING

            max_vox_percentage = 0.25
            x_size = int(round((x_max-x_min)/self.grid_size))
            y_size = int(round((y_max-y_min)/self.grid_size))
            z_size = int(round((z_max-z_min)/self.grid_size))

            max_num_vox = int(x_size*y_size*z_size*max_vox_percentage)

            self.gen_voxels = PointToVoxel(
                vsize_xyz = [self.grid_size, self.grid_size, self.grid_size],
                coors_range_xyz = roi.xyz,
                num_point_features = self.input_dim,
                max_num_voxels = max_num_vox,
                max_num_points_per_voxel = 4,
                device= torch.device('cuda')
            )
        else:
            self.roi = cfg.DATASET.RDR_SP_CUBE.ROI

            x_min, x_max = self.roi['x']
            y_min, y_max = self.roi['y']
            z_min, z_max = self.roi['z']
            self.min_roi = [x_min, y_min, z_min]

            self.grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE
            self.input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

            if self.cfg.DATASET.RDR_SP_CUBE.METHOD == 'quantile':
                self.type_data = 0
            else:
                print('* Exception error (Pre-processor): check RDR_SP_CUBE.METHOD')

    def forward(self, dict_item):
        if self.cfg_dataset_ver2:
            rdr_sparse = dict_item['rdr_sparse'].cuda()
            batch_indices = dict_item['batch_indices_rdr_sparse'].cuda()

            batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []

            for batch_idx in range(dict_item['batch_size']):
                corr_ind = torch.where(batch_indices == batch_idx)
                vox_in = rdr_sparse[corr_ind[0],:]
                
                voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(vox_in)
                voxel_batch_idx = torch.full((voxel_coords.shape[0], 1), batch_idx, device=rdr_sparse.device, dtype=torch.int64)
                voxel_coords = torch.cat((voxel_batch_idx, voxel_coords), dim=-1) # bzyx

                batch_voxel_features.append(voxel_features)
                batch_voxel_coords.append(voxel_coords)
                batch_num_pts_in_voxels.append(voxel_num_points)

            voxel_features, voxel_coords, voxel_num_points = torch.cat(batch_voxel_features), torch.cat(batch_voxel_coords), torch.cat(batch_num_pts_in_voxels)
            
            if self.is_with_simplified_pointnet:
                voxel_features = self.simplified_pointnet(voxel_features)
                if self.pooling_method == 'max':
                    voxel_features = torch.max(voxel_features, dim=1, keepdim=False)[0]
                elif self.pooling_method == 'mean':
                    voxel_features = voxel_features.sum(dim=1, keepdim=False)
                    normalizer = torch.clamp_min(voxel_num_points.view(-1,1), min=1.0).type_as(voxel_features)
                    voxel_features = voxel_features/normalizer
            else:
                voxel_features = voxel_features.sum(dim=1, keepdim=False)
                normalizer = torch.clamp_min(voxel_num_points.view(-1,1), min=1.0).type_as(voxel_features)
                voxel_features = voxel_features/normalizer

            dict_item['sp_features'] = voxel_features.contiguous()
            dict_item['sp_indices'] = voxel_coords.int()
            
        else:
            if self.type_data == 0:
                sp_cube = dict_item['rdr_sparse_cube'].cuda()
                B, N, C = sp_cube.shape

                list_batch_indices = []
                for batch_idx in range(B):
                    batch_indices = torch.full((N,1), batch_idx, dtype = torch.long)
                    list_batch_indices.append(batch_indices)
                sp_indices = torch.cat(list_batch_indices).cuda() # (N_1+...+N_B,1) -> 4=idx,z,y,x
                sp_cube = sp_cube.view(B*N, C) # (N_1+...+N_B,C)
                
                # Cut Doppler if self.input_dim = 4
                sp_cube = sp_cube[:,:self.input_dim]
                # print(sp_cube.shape)

                # Get z, y, x coord
                x_min, y_min, z_min = self.min_roi
                grid_size = self.grid_size
                x_coord, y_coord, z_coord = sp_cube[:, 0:1], sp_cube[:, 1:2], sp_cube[:, 2:3]

                z_ind = torch.ceil((z_coord-z_min) / grid_size).long()
                y_ind = torch.ceil((y_coord-y_min) / grid_size).long()
                x_ind = torch.ceil((x_coord-x_min) / grid_size).long() # -40.2 -> 0 for y

                sp_indices = torch.cat((sp_indices, z_ind, y_ind, x_ind), dim = -1)

                dict_item['sp_features'] = sp_cube
                dict_item['sp_indices'] = sp_indices

        return dict_item
