import torch
import torch.nn as nn
import numpy as np

class PillarsBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL.BACKBONE
        input_channels = self.model_cfg.IN_CHANNELS

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.roi = self.cfg.DATASET.LPC.ROI
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']
        pc_range = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
        voxel_size = np.array(self.cfg.MODEL.VOXEL_ENCODER.VOXEL_SIZE)
        grid_range = ((pc_range[3:]-pc_range[:3])/voxel_size).astype(int)
        # print(grid_range)
        self.nx, self.ny, self.nz = grid_range

        self.proj = nn.Linear(self.cfg.MODEL.VOXEL_ENCODER.NUM_POINT_FEATURES, self.cfg.MODEL.VOXEL_ENCODER.NUM_FILTERS[-1])
    def forward(self, data_dic):
        voxel_features, voxel_coords = data_dic['encoded_voxel_features'], data_dic['voxel_coords']
        voxel_features = self.proj(voxel_features)
        batch_spatial_features = []
        batch_size = data_dic['batch_size']
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                (self.cfg.MODEL.BACKBONE.IN_CHANNELS,
                (self.nz * self.nx * self.ny).astype(int).item()),
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            batch_mask = voxel_coords[:, 0] == batch_idx
            this_coords = voxel_coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = voxel_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        B, C, W, H = batch_size, self.cfg.MODEL.BACKBONE.IN_CHANNELS*self.nz, self.ny, self.nx
        B, C, W, H = int(B), int(C), int(W), int(H)

        spatial_features = batch_spatial_features.view(B, C, W, H) # twice larger than bev_feat of rdr
        # print(spatial_features.shape)
        data_dic['spatial_features'] = spatial_features

        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dic['spatial_features_2d'] = x # same shape as bev_feat for rdr
        # print(x.shape)

        return data_dic
