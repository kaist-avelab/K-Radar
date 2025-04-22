'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import torch.nn as nn

class RadarFiltProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarFiltProcessor, self).__init__()
        self.cfg = cfg

        self.roi = self.cfg.DATASET.RDR_FILT_SRT.ROI
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        self.list_roi = [x_min, y_min, z_min, x_max, y_max, z_max]
        self.vox_size_xyz = self.cfg.MODEL.PRE_PROCESSOR.VOX_SIZE_XYZ
        x_vox, y_vox, z_vox = self.vox_size_xyz
        self.spatial_size_zyx = [int((z_max-z_min)/z_vox), int((y_max-y_min)/y_vox), int((x_max-x_min)/x_vox)]

        # Doppler
        self.is_with_dop = self.cfg.MODEL.PRE_PROCESSOR.IS_WITH_DOP
        
        # filt type
        if self.cfg.DATASET.RDR_FILT_SRT.FILT_MODE == 'filt':
            self.type_filt = 0
            if self.is_with_dop:
                self.num_attr = 5 # X, Y, Z, PW, Dop
            else:
                self.num_attr = 4 # X, Y, Z, PW
        elif self.cfg.DATASET.RDR_FILT_SRT.FILT_MODE == 'both':
            self.type_filt = 1
            if self.is_with_dop:
                self.num_attr = 6 # X, Y, Z, PW, Dop, Flag
            else:
                self.num_attr = 5 # X, Y, Z, PW, Flag
        elif self.cfg.DATASET.RDR_FILT_SRT.FILT_MODE == 'cfar':
            self.type_filt = 2
            if self.is_with_dop:
                self.num_attr = 5 # X, Y, Z, PW, Dop
            else:
                self.num_attr = 4 # X, Y, Z, PW
        
        if self.cfg.MODEL.PRE_PROCESSOR.INPUT_DIM == 0:
            self.cfg.MODEL.PRE_PROCESSOR.INPUT_DIM = self.num_attr

    def forward(self, data_dic):
        rdr_filt_srt = data_dic['rdr_filt_srt'].cuda()
        batch_indices = data_dic['pts_batch_indices_rdr_filt_srt'].cuda()

        ### Filter mode ###
        if self.type_filt == 0: # filt
            rdr_filt_srt = rdr_filt_srt[:,:self.num_attr]
        elif self.type_filt == 1: # both X, Y, Z, PW, Dop, Flag
            rdr_filt_srt_flag = rdr_filt_srt[:,5:6]
            if self.is_with_dop:
                rdr_filt_srt = rdr_filt_srt[:,:5]
            else:
                rdr_filt_srt = rdr_filt_srt[:,:4]
            # print(rdr_filt_srt_flag.shape)
            # print(torch.unique(rdr_filt_srt_flag))
            rdr_filt_srt = torch.concat((rdr_filt_srt, rdr_filt_srt_flag), dim=1)
            # print(rdr_filt_srt.shape)
        elif self.type_filt == 2:
            rdr_filt_srt = rdr_filt_srt[:,:self.num_attr]
        ### Filter mode ###

        # print(self.vox_size)
        # print(self.list_roi)
        # print(rdr_filt_srt.shape)       # (N, self.num_attr)
        # print(batch_indices.shape)      # (N)
        # exit()

        x_min, y_min, z_min, x_max, y_max, z_max = self.list_roi
        x_coord, y_coord, z_coord = rdr_filt_srt[:,0:1], rdr_filt_srt[:,1:2], rdr_filt_srt[:,2:3]
        # print(torch.min(x_coord), torch.max(x_coord))
        # print(torch.min(y_coord), torch.max(y_coord))
        # print(torch.min(z_coord), torch.max(z_coord))
        # print(x_min, y_min, z_min, x_max, y_max, z_max)
        # exit()

        # using floor & check max idx
        x_vox, y_vox, z_vox = self.vox_size_xyz
        z_shape, y_shape, x_shape = self.spatial_size_zyx
        z_ind = torch.clamp(torch.floor((z_coord-z_min) / z_vox).int(), 0, z_shape-1)
        y_ind = torch.clamp(torch.floor((y_coord-y_min) / y_vox).int(), 0, y_shape-1)
        x_ind = torch.clamp(torch.floor((x_coord-x_min) / x_vox).int(), 0, x_shape-1)

        # print(self.spatial_size_zyx)

        sp_indices = torch.cat((batch_indices.unsqueeze(-1).int(), z_ind, y_ind, x_ind), dim=-1)

        data_dic['sp_features'] = rdr_filt_srt
        data_dic['sp_indices'] = sp_indices

        return data_dic
        