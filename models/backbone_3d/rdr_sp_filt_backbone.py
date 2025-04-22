'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

class RadarSparseFiltBackbone(nn.Module):
    def __init__(self, cfg):
        super(RadarSparseFiltBackbone, self).__init__()
        self.cfg = cfg

        self.IS_GET_ATTN_WEIGHTS = True # False

        self.roi = self.cfg.DATASET.RDR_FILT_SRT.ROI
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        self.list_roi = [x_min, y_min, z_min, x_max, y_max, z_max]
        self.vox_size_xyz = self.cfg.MODEL.PRE_PROCESSOR.VOX_SIZE_XYZ
        x_vox, y_vox, z_vox = self.vox_size_xyz

        z_shape = int((z_max-z_min) / z_vox)
        y_shape = int((y_max-y_min) / y_vox)
        x_shape = int((x_max-x_min) / x_vox)
        self.spatial_size_zyx = [int((z_max-z_min)/z_vox), int((y_max-y_min)/y_vox), int((x_max-x_min)/x_vox)]

        # print(self.spatial_shape)

        cfg_model = self.cfg.MODEL
        input_dim = cfg_model.PRE_PROCESSOR.INPUT_DIM # 4

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

        # to BEV
        self.type_to_bev = cfg_model.BACKBONE.TO_BEV.TYPE

        # transpose
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING

        # mha
        self.n_heads = cfg_model.BACKBONE.TO_BEV.MHA.N_HEADS

        if self.type_to_bev == 'res':
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] # in [64, 128, 256] / ENCODING.CHANNEL
                setattr(self, f'in_patch_format{idx_bev}', Rearrange('b c z y x -> (b y x) z c'))
                # z_size = int(z_shape/(2**idx_bev))
                y_size = int(y_shape/(2**idx_bev))
                x_size = int(x_shape/(2**idx_bev))
                setattr(self, f'q_dim_reduction{idx_bev}', nn.Parameter(torch.randn(1, y_size, x_size, temp_enc_ch))) # dim_query = temp_enc_channel
                setattr(self, f'to_query{idx_bev}', Rearrange('b y x (n_q c) -> (b y x) n_q c', n_q=1))
                setattr(self, f'mha{idx_bev}', nn.MultiheadAttention(temp_enc_ch, self.n_heads, 0., batch_first=True))
                setattr(self, f'to_bev{idx_bev}', \
                    Rearrange('(b y x) n_q (c n_stride_y n_stride_x) -> b (n_q c) (y n_stride_y) (x n_stride_x)', \
                    y=y_size, x=x_size, n_stride_y=(2**idx_bev), n_stride_x=(2**idx_bev)))
        elif self.type_to_bev == 'z_embed':
            self.is_z_embed = True
            for idx_bev in range(self.num_layer):
                setattr(self, f'chzcat{idx_bev}', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
        elif self.type_to_bev == 'ori':
            self.is_z_embed = False
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] # in [64, 128, 256] / ENCODING.CHANNEL
                temp_out_channel = list_bev_channel[idx_bev]
                z_kernel_size = int(z_shape/(2**idx_bev))
                # print(z_kernel_size)
                setattr(self, f'toBEV{idx_bev}', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
        
        # activation
        self.relu = nn.ReLU()

    def forward(self, dict_item):
        sparse_features, sparse_indices = dict_item['sp_features'], dict_item['sp_indices']

        input_sp_tensor = spconv.SparseConvTensor(
            features=sparse_features,
            indices=sparse_indices,
            spatial_shape=self.spatial_size_zyx,
            batch_size=dict_item['batch_size']
        )

        # print(input_sp_tensor.dense().shape)
        x = self.input_conv(input_sp_tensor)
        # print(x.dense().shape)

        list_bev_features = []

        if self.IS_GET_ATTN_WEIGHTS:
            dict_att = dict()
            dict_att['roi'] = self.roi
            dict_att['vox'] = self.vox_size_xyz

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

            if self.type_to_bev == 'res':
                kv_bev = getattr(self, f'in_patch_format{idx_layer}')(x.dense())
                q_bev = getattr(self, f'q_dim_reduction{idx_layer}')
                q_bev = repeat(q_bev, '() y x c -> b y x c', b = dict_item['batch_size'])
                q_bev = getattr(self, f'to_query{idx_layer}')(q_bev)
                
                if self.IS_GET_ATTN_WEIGHTS:
                    # print(q_bev.shape)  # 5760, 1,  64
                    # print(kv_bev.shape) # 5760, 20, 64
                    mha_bev_with_weights = getattr(self, f'mha{idx_layer}')(q_bev, kv_bev, kv_bev)
                    mha_bev = mha_bev_with_weights[0]
                    mha_bev_with_weights = mha_bev_with_weights[1]
                    # print(mha_bev.shape)                # 5760, 1,  64
                    # print(mha_bev_with_weights.shape)   # 5760, 1,  20

                    # setattr(self, f'to_bev{idx_bev}', \
                    # Rearrange('(b y x) n_q (c n_stride_y n_stride_x) -> b (n_q c) (y n_stride_y) (x n_stride_x)', \
                    # y=y_size, x=x_size, n_stride_y=(2**idx_bev), n_stride_x=(2**idx_bev)))

                    z_sh, y_sh, x_sh = self.spatial_size_zyx
                    mha_bev_weights_in_bev_layer = rearrange(mha_bev_with_weights, '(b y x) n_q z -> b n_q z y x', \
                                                    b=dict_item['batch_size'], y=int(y_sh/2**idx_layer), x=int(x_sh/2**idx_layer))
                    mha_bev_weights_in_bev_layer = mha_bev_weights_in_bev_layer.squeeze(1)
                    dict_att[f'att{idx_layer}'] = mha_bev_weights_in_bev_layer
                    # print(mha_bev_weights_in_bev_layer.shape)
                else:
                    mha_bev = getattr(self, f'mha{idx_layer}')(q_bev, kv_bev, kv_bev)[0] # attn, 1 is weight
                # print(len(mha_bev))
                # print(mha_bev.shape)
                bev_dense = getattr(self, f'to_bev{idx_layer}')(mha_bev)
                # print(bev_dense.shape)

            elif self.type_to_bev == 'z_embed':
                bev_dense = getattr(self, f'chzcat{idx_layer}')(x.dense())
                bev_dense = getattr(self, f'convtrans2d{idx_layer}')(bev_dense)
                bev_dense = getattr(self, f'bnt{idx_layer}')(bev_dense)
                bev_dense = self.relu(bev_dense)
            elif self.type_to_bev == 'ori':
                bev_sp = getattr(self, f'toBEV{idx_layer}')(x)
                bev_sp = bev_sp.replace_feature(getattr(self, f'bnBEV{idx_layer}')(bev_sp.features))
                bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
                # print(bev_sp.dense().shape)

                # B, C, 1, Y/st, X/st -> B, C, Y, X
                bev_dense = getattr(self, f'convtrans2d{idx_layer}')(bev_sp.dense().squeeze(2))
                bev_dense = getattr(self, f'bnt{idx_layer}')(bev_dense)
                bev_dense = self.relu(bev_dense)

            list_bev_features.append(bev_dense)

        if self.IS_GET_ATTN_WEIGHTS:
            dict_item['att'] = dict_att

        bev_features = torch.cat(list_bev_features, dim = 1)
        # print(bev_features.shape)
        dict_item['bev_feat'] = bev_features # B, C, Y, X

        return dict_item
