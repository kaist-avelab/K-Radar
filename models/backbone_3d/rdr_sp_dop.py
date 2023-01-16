'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from einops.layers.torch import Rearrange

class ParallelEncoder(nn.Module):
    def __init__(self, cfg):
        super(ParallelEncoder, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        z_shape = int((z_max-z_min) / grid_size)
        y_shape = int((y_max-y_min) / grid_size)
        x_shape = int((x_max-x_min) / grid_size)

        self.spatial_shape = [z_shape, y_shape, x_shape]

        cfg_model = self.cfg.MODEL
        input_dim = 4

        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        list_enc_padding = cfg_model.BACKBONE.ENCODING.PADDING
        list_enc_stride  = cfg_model.BACKBONE.ENCODING.STRIDE
        
        # 1x1 conv / 4->ENCODING.CHANNEL[0]
        self.input_conv = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        
        # encoder
        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = list_enc_channel[0] # in [64, 128, 256]
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] # in [64, 128, 256]
            temp_ch = list_enc_channel[idx_enc]
            temp_pd = list_enc_padding[idx_enc]
            setattr(self, f'spconv{idx_enc}', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}a', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}a', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}b', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}b', nn.BatchNorm1d(temp_ch))
        
        # activation
        self.relu = nn.ReLU()

    def forward(self, sparse_features, sparse_indices, batch_size):
        input_sp_tensor = spconv.SparseConvTensor(
            features=sparse_features,
            indices=sparse_indices.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )
        x = self.input_conv(input_sp_tensor)

        list_enc_sp_features = []
        for idx_layer in range(self.num_layer):
            # print(idx_layer)
            x = getattr(self, f'spconv{idx_layer}')(x)
            x = x.replace_feature(getattr(self, f'bn{idx_layer}')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'subm{idx_layer}a')(x)
            x = x.replace_feature(getattr(self, f'bn{idx_layer}a')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'subm{idx_layer}b')(x)
            x = x.replace_feature(getattr(self, f'bn{idx_layer}b')(x.features))
            x = x.replace_feature(self.relu(x.features))
            # print(x.dense().shape)
            list_enc_sp_features.append(x)

        return list_enc_sp_features

class RadarSparseBackboneDop(nn.Module):
    def __init__(self, cfg):
        super(RadarSparseBackboneDop, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        z_shape = int((z_max-z_min) / grid_size)
        y_shape = int((y_max-y_min) / grid_size)
        x_shape = int((x_max-x_min) / grid_size)

        self.spatial_shape = [z_shape, y_shape, x_shape]

        self.enc_pw = ParallelEncoder(cfg=self.cfg)
        self.enc_dop = ParallelEncoder(cfg=self.cfg)

        # Required parameters for high semantic concat
        cfg_model = self.cfg.MODEL
        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        self.num_layer = len(list_enc_channel)
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL

        # to BEV
        if cfg_model.BACKBONE.TO_BEV.IS_Z_EMBED:
            self.is_z_embed = True
            for idx_bev in range(self.num_layer):
                setattr(self, f'chzcat_pw{idx_bev}', Rearrange('b c z y x -> b (c z) y x'))
                setattr(self, f'chzcat_dop{idx_bev}', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev)*2) # *2 for pw/dop concat
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
        else:
            self.is_z_embed = False
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] # in [64, 128, 256] / ENCODING.CHANNEL
                temp_out_channel = list_bev_channel[idx_bev]
                z_kernel_size = int(z_shape/(2**idx_bev))
                # print(z_kernel_size)
                setattr(self, f'toBEV_pw{idx_bev}', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                # print(z_kernel_size)
                setattr(self, f'toBEV_dop{idx_bev}', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV_pw{idx_bev}', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'bnBEV_dop{idx_bev}', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
        
        # activation
        self.relu = nn.ReLU()

    def forward(self, dict_item):
        sp_features_pw = dict_item['sp_features_pw']
        sp_features_dop = dict_item['sp_features_dop']
        sp_indices = dict_item['sp_indices']
        batch_size = dict_item['batch_size']

        list_enc_sp_features_pw = self.enc_pw(sp_features_pw, sp_indices, batch_size)
        list_enc_sp_features_dop = self.enc_dop(sp_features_dop, sp_indices, batch_size)
        
        list_bev_features = []
        for idx_layer in range(self.num_layer):
            enc_sp_pw = list_enc_sp_features_pw[idx_layer]
            enc_sp_dop = list_enc_sp_features_dop[idx_layer]

            if self.is_z_embed:
                bev_dense_pw = getattr(self, f'chzcat_pw{idx_layer}')(enc_sp_pw.dense())
                bev_dense_dop = getattr(self, f'chzcat_dop{idx_layer}')(enc_sp_dop.dense())
                bev_dense = getattr(self, f'convtrans2d{idx_layer}')(torch.cat((bev_dense_pw, bev_dense_dop), dim=1))
            else:
                bev_sp_pw = getattr(self, f'toBEV_pw{idx_layer}')(enc_sp_pw)
                bev_sp_pw = bev_sp_pw.replace_feature(getattr(self, f'bnBEV_pw{idx_layer}')(bev_sp_pw.features))
                bev_sp_pw = bev_sp_pw.replace_feature(self.relu(bev_sp_pw.features))

                bev_sp_dop = getattr(self, f'toBEV_dop{idx_layer}')(enc_sp_dop)
                bev_sp_dop = bev_sp_dop.replace_feature(getattr(self, f'bnBEV_dop{idx_layer}')(bev_sp_dop.features))
                bev_sp_dop = bev_sp_dop.replace_feature(self.relu(bev_sp_dop.features))

                # B, C, 1, Y/st, X/st -> B, C, Y, X
                bev_dense = getattr(self, f'convtrans2d{idx_layer}')(torch.cat((bev_sp_pw.dense().squeeze(2), bev_sp_dop.dense().squeeze(2)), dim=1))

            bev_dense = getattr(self, f'bnt{idx_layer}')(bev_dense)
            bev_dense = self.relu(bev_dense)
            list_bev_features.append(bev_dense)

        # print(bev_features.shape)
        bev_features = torch.cat(list_bev_features, dim = 1)
        dict_item['bev_feat'] = bev_features # B, C, Y, X

        return dict_item
