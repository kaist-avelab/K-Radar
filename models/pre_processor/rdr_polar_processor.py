'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from einops import rearrange

class RadarPolarProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarPolarProcessor, self).__init__()
        self.cfg = cfg

        input_mode_str = cfg.MODEL.PRE_PROCESSOR.INPUT_MODE
        if input_mode_str == 'rae_xyz_pw':
            self.input_mode = 0
        elif input_mode_str == 'rae_xyz_pw_dop':
            self.input_mode = 1

        ### Get arr DRAE from cfg ###
        self.arr_range = self.cfg.DATASET.SP_RDR_TESSERACT.ARR_RANGE
        self.arr_azimuth = self.cfg.DATASET.SP_RDR_TESSERACT.ARR_AZIMUTH
        self.arr_elevation = self.cfg.DATASET.SP_RDR_TESSERACT.ARR_ELEVATION
        self.arr_doppler = self.cfg.DATASET.SP_RDR_TESSERACT.ARR_DOPPLER

        self.spatial_shape = [len(self.arr_range), len(self.arr_azimuth), len(self.arr_elevation)]

        # print(self.arr_range, self.arr_azimuth, self.arr_elevation, self.arr_doppler)
        # print(len(self.arr_range), len(self.arr_azimuth), len(self.arr_elevation), len(self.arr_doppler))
        ### Get arr DRAE from cfg ###

    def forward(self, dict_item):
        # 0, 1, 2, 3, 4, 5, 6,  7,   8,     9,     10
        # r, a, e, x, y, z, pw, dop, idx_r, idx_a, idx_e
        # print(dict_item['sp_rdr_tesseract'].shape) # 4, 66560, 11

        if self.input_mode == 0:
            sp_rdr_tesseract = dict_item['sp_rdr_tesseract']
            B, N, _ = sp_rdr_tesseract.shape
            sp_features = sp_rdr_tesseract[:,:,:7]
            sp_features_idx = sp_rdr_tesseract[:,:,8:].int() # ind_r, ind_a, ind_e
            
            list_batch_indices = []
            for idx_b in range(B):
                batch_indices = torch.full((N,1), idx_b, dtype=torch.int32)
                list_batch_indices.append(batch_indices)
            sp_indices = torch.cat(list_batch_indices, dim=0)
            sp_indices = torch.cat((sp_indices, rearrange(sp_features_idx, 'b n c -> (b n) c')), dim=-1)

            input_sp_tensor = spconv.SparseConvTensor(
                features=rearrange(sp_features, 'b n c -> (b n) c').cuda(),
                indices=sp_indices.cuda(),
                spatial_shape=self.spatial_shape,
                batch_size=B
            )

            dict_item['sp_tensor'] = input_sp_tensor # 4, 7, 200, 104, 32
            # print(input_sp_tensor.dense().shape)

        return dict_item
