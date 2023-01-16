'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn

# V2
class RadarSparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarSparseProcessor, self).__init__()
        self.cfg = cfg
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

            # print(torch.min(z_coord), torch.max(z_coord))
            # print(torch.min(y_coord), torch.max(y_coord))
            # print(torch.min(x_coord), torch.max(x_coord))

            z_ind = torch.ceil((z_coord-z_min) / grid_size).long()
            y_ind = torch.ceil((y_coord-y_min) / grid_size).long()
            x_ind = torch.ceil((x_coord-x_min) / grid_size).long() # -40.2 -> 0 for y

            # print(torch.min(z_ind), torch.max(z_ind))
            # print(torch.min(y_ind), torch.max(y_ind))
            # print(torch.min(x_ind), torch.max(x_ind))

            sp_indices = torch.cat((sp_indices, z_ind, y_ind, x_ind), dim = -1)

            dict_item['sp_features'] = sp_cube
            dict_item['sp_indices'] = sp_indices

        return dict_item
