import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import spconv.pytorch as spconv # TODO: Change this one into MinkowskiEngine
from spconv.pytorch.utils import PointToVoxel

try:
    from utils import box_coder_utils, common_utils, loss_utils
except:
    import sys
    import os.path as osp
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from utils import box_coder_utils, common_utils, loss_utils

from models.head.target_assigner.anchor_generator import AnchorGenerator
from models.head.target_assigner.atss_target_assigner import ATSSTargetAssigner
from models.head.target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner

from models.model_utils import model_nms_utils
from ops.iou3d_nms import iou3d_nms_utils

### ref: Spiking PointNet (https://github.com/DayongRen/Spiking-PointNet/tree/main) ###
class SpikeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        if self._spiking is not True: # and len(x.shape) == 4:
            x = x.mean([0])
        return x

def spike_activation(x, ste=False, temp=5.0):
    out_s = torch.gt(x, 0.5)
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp

def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y

def mem_update(x_in, mem, V_th, decay, grad_scale=1., temp=5.0):
    mem = mem * decay + x_in
    spike = spike_activation(mem / V_th, temp=temp)
    mem = mem * (1 - spike)
    return mem, spike

class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self, step, temp):
        super(LIFAct, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = temp
        self.grad_scale = 0.1

    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.rand_like(x[0]) * 0.5
        out = []
        for i in range(self.step):
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out
    
class SpikeConv(SpikeModule):
    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.conv(x)
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out

class SpikeBatchNorm(SpikeModule):
    def __init__(self, BN: nn.BatchNorm2d, step=2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(BN.num_features)
        self.step = step
    
    def forward(self, x):
        if self._spiking is not True:
            return self.BN(x)
        
        tensor_size = x.size()
        len_dim = len(tensor_size)
        changed_idx = list(range(1, len_dim)) + [0]
        
        out = x.permute(*changed_idx).contiguous()
        ret_size = out.size()
        out = out.view(tensor_size[1], tensor_size[2], -1)
        out = self.bn1(out)
        out = out.view(*ret_size)
        out = out.permute(*([len_dim - 1] + list(range(len_dim - 1)))).contiguous()
        
        return out
### ref: Spiking PointNet (https://github.com/DayongRen/Spiking-PointNet/tree/main) ###

class Voxelizer(nn.Module):
    def __init__(self, cfg):
        super(Voxelizer, self).__init__()
        self.cfg = cfg
        
        cfg_ds = self.cfg.DATASET
        roi = cfg_ds.roi
        x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
        self.min_roi = [x_min, y_min, z_min]
        self.vox_xyz = roi.voxel_size
        x_vox, y_vox, z_vox = self.vox_xyz
        self.input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

        x_shape = int(round((x_max-x_min)/x_vox))
        y_shape = int(round((y_max-y_min)/y_vox))
        z_shape = int(round((z_max-z_min)/z_vox))

        self.gen_voxels = PointToVoxel(
            vsize_xyz = self.vox_xyz,
            coors_range_xyz = roi.xyz,
            num_point_features = self.input_dim,
            max_num_voxels = x_shape*y_shape*z_shape,
            max_num_points_per_voxel = 1,
            device=torch.device('cuda')
        )

    def forward(self, dict_item):
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
        
        sparse_features = (torch.squeeze(voxel_features, dim=1)).contiguous()
        sparse_indices = voxel_coords.int()
        sp_tensor = spconv.SparseConvTensor(
            features=sparse_features,
            indices=sparse_indices.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        dict_item['dense_input'] = sp_tensor.dense()

        return dict_item

class RtnhDenseBackbone(nn.Module):
    def __init__(self, cfg):
        super(RtnhDenseBackbone, self).__init__()
        self.cfg = cfg
        
        ### Params for voxelization ###
        cfg_ds = self.cfg.DATASET
        roi = cfg_ds.roi
        x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
        self.min_roi = [x_min, y_min, z_min]
        self.vox_xyz = roi.voxel_size
        x_vox, y_vox, z_vox = self.vox_xyz
        self.input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

        x_shape = int(round((x_max-x_min)/x_vox))
        y_shape = int(round((y_max-y_min)/y_vox))
        z_shape = int(round((z_max-z_min)/z_vox))
        ### Params for voxelization ###

        ### Backbone ###
        self.spatial_shape = [z_shape, y_shape, x_shape]
        cfg_model = self.cfg.MODEL
        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        list_enc_padding = cfg_model.BACKBONE.ENCODING.PADDING
        list_enc_stride  = cfg_model.BACKBONE.ENCODING.STRIDE

        self.use_3_layer = cfg_model.BACKBONE.get('USE_3_LAYER', False)

        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = self.input_dim
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] # [64, 128, 256]
            temp_ch = list_enc_channel[idx_enc]
            temp_st = list_enc_stride[idx_enc]
            temp_pd = list_enc_padding[idx_enc]

            # Default 3 layers -> 2 layers (due to required large memory dense conv)
            setattr(self, f'conv0_{idx_enc}', nn.Conv3d(temp_in_ch, temp_ch, 3, temp_st, temp_pd))
            setattr(self, f'bn0_{idx_enc}', nn.BatchNorm3d(temp_ch))
            setattr(self, f'relu0_{idx_enc}', nn.ReLU())    # added for LIFAct
            setattr(self, f'conv1_{idx_enc}', nn.Conv3d(temp_ch, temp_ch, 3, 1, 1))
            setattr(self, f'bn1_{idx_enc}', nn.BatchNorm3d(temp_ch))
            setattr(self, f'relu1_{idx_enc}', nn.ReLU())    # added for LIFAct

            if self.use_3_layer:
                setattr(self, f'conv2_{idx_enc}', nn.Conv3d(temp_ch, temp_ch, 3, 1, 1))
                setattr(self, f'bn2_{idx_enc}', nn.BatchNorm3d(temp_ch))
                setattr(self, f'relu2_{idx_enc}', nn.ReLU())    # added for LIFAct
        
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING

        for idx_bev in range(self.num_layer):
            temp_enc_ch = list_enc_channel[idx_bev] # in [64, 128, 256]
            temp_out_channel = list_bev_channel[idx_bev]
            z_kernel_size = int(z_shape/(2**idx_bev))

            setattr(self, f'toBEV_{idx_bev}', nn.Conv3d(temp_enc_ch, temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
            setattr(self, f'bnBEV_{idx_bev}', nn.BatchNorm2d(temp_enc_ch))
            setattr(self, f'reluBEV0_{idx_bev}', nn.ReLU())
            setattr(self, f'convtrans2d{idx_bev}', nn.ConvTranspose2d(temp_enc_ch, temp_out_channel, \
                    kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
            setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
            setattr(self, f'reluBEV1_{idx_bev}', nn.ReLU())
        ### Backbone ###
        
        ### Head ###
        self.cfg_head = self.cfg.MODEL.HEAD
        point_cloud_range = roi.xyz
        grid_size = np.array([x_shape, y_shape, z_shape], dtype=np.int64)

        # class
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)

        anchor_target_cfg = self.cfg_head.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.cfg_head.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        input_channels = self.cfg_head.INPUT_CHANNELS
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.cfg_head.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.cfg_head.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        ### Head ###
        
        self.init_weights()

        self._spiking = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def measure_spiking_rate(self, x):
        list_len = list(x.shape)
        shape_len = 1
        for temp in list_len:
            shape_len *= temp
        spiking_rate = (shape_len-len(torch.where(x==0.)[0]))/shape_len
        return spiking_rate

    def forward(self, dict_item, is_get_spiking_rate=False):
        ### Backbone ###
        # TODO: removing sparse tensors -> change voxleization function
        x = dict_item['dense_input']

        if is_get_spiking_rate:
            list_spiking_rate = []

        list_bev_features = []
        for idx_layer in range(self.num_layer):
            x = getattr(self, f'conv0_{idx_layer}')(x)
            x = getattr(self, f'bn0_{idx_layer}')(x)
            x = getattr(self, f'relu0_{idx_layer}')(x)

            if is_get_spiking_rate:
                list_spiking_rate.append(self.measure_spiking_rate(x))

            x = getattr(self, f'conv1_{idx_layer}')(x)
            x = getattr(self, f'bn1_{idx_layer}')(x)
            x = getattr(self, f'relu1_{idx_layer}')(x)

            if is_get_spiking_rate:
                list_spiking_rate.append(self.measure_spiking_rate(x))

            if self.use_3_layer:
                x = getattr(self, f'conv2_{idx_layer}')(x)
                x = getattr(self, f'bn2_{idx_layer}')(x)
                x = getattr(self, f'relu2_{idx_layer}')(x)

                if is_get_spiking_rate:
                    list_spiking_rate.append(self.measure_spiking_rate(x))

            bev_feat = getattr(self, f'toBEV_{idx_layer}')(x)
            if self._spiking:
                bev_feat = torch.squeeze(bev_feat, dim=3)
            else:
                bev_feat = torch.squeeze(bev_feat, dim=2)
            bev_feat = getattr(self, f'bnBEV_{idx_layer}')(bev_feat)
            bev_feat = getattr(self, f'reluBEV0_{idx_layer}')(bev_feat)
            
            if is_get_spiking_rate:
                list_spiking_rate.append(self.measure_spiking_rate(bev_feat))
            
            bev_feat = getattr(self, f'convtrans2d{idx_layer}')(bev_feat)
            bev_feat = getattr(self, f'bnt{idx_layer}')(bev_feat)
            bev_feat = getattr(self, f'reluBEV1_{idx_layer}')(bev_feat)

            if is_get_spiking_rate:
                list_spiking_rate.append(self.measure_spiking_rate(bev_feat))

            list_bev_features.append(bev_feat)

        if self._spiking:
            bev_feat = torch.cat(list_bev_features, dim=2)
        else:
            bev_feat = torch.cat(list_bev_features, dim=1)
        ### Backbone ###
        
        ### Head ###
        cls_preds = self.conv_cls(bev_feat)
        box_preds = self.conv_box(bev_feat)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(bev_feat)
            dict_item['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        
        dict_item['cls_preds'] = cls_preds
        dict_item['box_preds'] = box_preds
        ### Head ###

        if is_get_spiking_rate:
            dict_item['spiking_rate'] = np.mean(list_spiking_rate)
        
        return dict_item
    
    def generate_anchors(self, anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

class SpikingRTNH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.step = cfg.MODEL.SPIKING.STEP
        self.temp = cfg.MODEL.SPIKING.TEMP
        
        gradual_spiking = cfg.MODEL.SPIKING.get('GRADUAL_SPIKING', None)
        if gradual_spiking is not None:
            self.is_gradual = gradual_spiking.IS_GRADUAL
            self.gradual_portion = gradual_spiking.PORTION

        ### Params for voxelization ###
        cfg_ds = self.cfg.DATASET
        roi = cfg_ds.roi
        x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
        self.min_roi = [x_min, y_min, z_min]
        self.vox_xyz = roi.voxel_size
        x_vox, y_vox, z_vox = self.vox_xyz
        self.input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

        x_shape = int(round((x_max-x_min)/x_vox))
        y_shape = int(round((y_max-y_min)/y_vox))
        z_shape = int(round((z_max-z_min)/z_vox))
        self.spatial_shape = [z_shape, y_shape, x_shape]

        self.gen_voxels = PointToVoxel(
            vsize_xyz = self.vox_xyz,
            coors_range_xyz = roi.xyz,
            num_point_features = self.input_dim,
            max_num_voxels = x_shape*y_shape*z_shape,
            max_num_points_per_voxel = 1,
            device= torch.device('cuda')
        )
        ### Params for voxelization ###

        ### Backbone ###
        self.backbone = RtnhDenseBackbone(cfg) # Note that 1 by 1 conv for head is included in the backbone
        self.spike_module_refactor(self.backbone, step=self.step, temp=self.temp)
        ### Backbone ###

        ### Head ###
        self.cfg_head = self.cfg.MODEL.HEAD
        point_cloud_range = roi.xyz
        grid_size = np.array([x_shape, y_shape, z_shape], dtype=np.int64)

        # class
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)

        anchor_target_cfg = self.cfg_head.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.cfg_head.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.cfg_head.LOSS_CONFIG)

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING
        self.use_multihead = False
        ### Head ###

        self.spiking = True
        if self.spiking:
            self.backbone._spiking = True
            self.set_spike_state(True)

    def spike_module_refactor(self, module: nn.Module, step=2, temp=3.0):
        """
        Recursively replace the normal conv1d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():
            if (isinstance(child_module, nn.Conv1d)) or \
                (isinstance(child_module, nn.Conv2d)) or \
                (isinstance(child_module, nn.Conv3d)) or \
                (isinstance(child_module, nn.ConvTranspose2d)):
                setattr(module, name, SpikeConv(child_module, step=step))

            elif isinstance(child_module, nn.ReLU):
                setattr(module, name, LIFAct(step=step, temp=temp))

            elif (isinstance(child_module, nn.BatchNorm1d)) or \
                (isinstance(child_module, nn.BatchNorm2d)) or \
                (isinstance(child_module, nn.BatchNorm3d)):
                setattr(module, name, SpikeBatchNorm(child_module, step=step))
            
            else:
                self.spike_module_refactor(child_module, step=step, temp=temp)

    def forward(self, dict_item, is_get_spiking_rate=False):
        rdr_sparse = dict_item['rdr_sparse'].cuda()
        batch_indices = dict_item['batch_indices_rdr_sparse'].cuda()
        
        if self.is_gradual:
            if self.spiking:
                list_dense_input = []
                for idx_step in range(self.step):
                    batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []

                    for batch_idx in range(dict_item['batch_size']):
                        corr_ind = torch.where(batch_indices == batch_idx)
                        vox_in = rdr_sparse[corr_ind[0],:]

                        if idx_step == 0:
                            vox_temp = vox_in
                        else:
                            gradual_portion = (self.gradual_portion)**idx_step
                            valid_idx = torch.where(vox_in[:,3]>torch.quantile(vox_in[:,3], 1-gradual_portion))[0]
                            vox_temp = vox_in[valid_idx,:]
                        
                        voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(vox_temp)
                        voxel_batch_idx = torch.full((voxel_coords.shape[0], 1), batch_idx, device=rdr_sparse.device, dtype=torch.int64)
                        voxel_coords = torch.cat((voxel_batch_idx, voxel_coords), dim=-1) # bzyx

                        batch_voxel_features.append(voxel_features)
                        batch_voxel_coords.append(voxel_coords)
                        batch_num_pts_in_voxels.append(voxel_num_points)

                    voxel_features, voxel_coords, voxel_num_points = torch.cat(batch_voxel_features), torch.cat(batch_voxel_coords), torch.cat(batch_num_pts_in_voxels)
                    
                    sp_tensor = spconv.SparseConvTensor(
                        features=torch.squeeze(voxel_features, dim=1),
                        indices=voxel_coords.int(),
                        spatial_shape=self.spatial_shape,
                        batch_size=dict_item['batch_size']
                    )
                    list_dense_input.append(sp_tensor.dense())

                dict_item['dense_input'] = torch.stack(list_dense_input)
        else:
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
            
            sp_tensor = spconv.SparseConvTensor(
                features=torch.squeeze(voxel_features, dim=1),
                indices=voxel_coords.int(),
                spatial_shape=self.spatial_shape,
                batch_size=dict_item['batch_size']
            )
            dict_item['dense_input'] = sp_tensor.dense()

            if self.spiking:
                dict_item['dense_input'] = dict_item['dense_input'].repeat(self.step, 1, 1, 1, 1, 1) # T, B, C, Z, Y, X
        
        dict_item = self.backbone(dict_item, is_get_spiking_rate)

        dict_item['gt_boxes'] = dict_item['gt_boxes'].cuda()

        cls_preds = dict_item['cls_preds']
        box_preds = dict_item['box_preds']

        if 'dir_cls_preds' in dict_item:
            dir_cls_preds = dict_item['dir_cls_preds']

        if self.spiking:
            cls_preds = cls_preds.mean([0])
            box_preds = box_preds.mean([0])
            if 'dir_cls_preds' in dict_item:
                dir_cls_preds = dir_cls_preds.mean([0])

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if 'dir_cls_preds' in dict_item:
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=dict_item['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        else:
        # if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=dict_item['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            dict_item['batch_cls_preds'] = batch_cls_preds
            dict_item['batch_box_preds'] = batch_box_preds
            dict_item['cls_preds_normalized'] = False

            dict_item = self.post_processing(dict_item)

        return dict_item

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike
        for m in self.backbone.modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(use_spike)

    def set_spike_before(self, name):
        self.set_spike_state(False)
        for n, m in self.backbone.named_modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(True)
            if name == n:
                break

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.cfg_head,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.cfg_head.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.cfg_head.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.cfg_head.DIR_OFFSET,
                num_bins=self.cfg_head.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.cfg_head.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.cfg_head.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def loss(self, dict_item):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()

        if self.is_logging:
            dict_item['logging'] = dict()
            dict_item['logging'].update(tb_dict)

        return rpn_loss
    
    def logging_dict_loss(self, loss, name_key):
        try:
            log_loss = loss.cpu().detach().item()
        except:
            log_loss = loss # for 0. loss

        return {name_key: log_loss}

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.cfg_head.DIR_OFFSET
            dir_limit_offset = self.cfg_head.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.cfg_head.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds
    
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.cfg_head.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1 
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                    
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )        

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

            batch_dict['pred_dicts'] = pred_dicts
            batch_dict['recall_dict'] = recall_dict

        return batch_dict
    
    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
    

if __name__ == '__main__':
    from utils.util_config import cfg, cfg_from_yaml_file

    ### Check batch norm equivalency ###
    # (1) ANN case
    # temp_tensor = torch.randn(2, 64, 10, 20, 30)*50

    # init_weight = torch.randn(64) # gamma
    # init_bias = torch.randn(64) # beta

    # bn3 = nn.BatchNorm3d(64)
    # print(bn3.weight.data)
    # print(bn3.bias.data)
    # bn3.weight.data = init_weight
    # bn3.bias.data = init_bias
    # print(bn3.weight.data)
    # print(bn3.bias.data)

    # bn1 = nn.BatchNorm1d(64)
    # bn1.weight.data = init_weight
    # bn1.bias.data = init_bias
    # print(bn1.weight.data)
    # print(bn1.bias.data)

    # # from bn3
    # out3 = bn3(temp_tensor)

    # tensor_size = temp_tensor.size()
    # out1 = temp_tensor.view(tensor_size[0], tensor_size[1], -1)
    # out1 = bn1(out1)
    # out1 = out1.view(tensor_size)

    # print(out3.shape)
    # print(out1.shape)
    # print(out3)
    # print(out1)
    # print(out3==out1)

    # (2) SNN case
    # temp_tensor = torch.randn(8, 2, 64, 10, 20, 30)*50 # T, B, C, Z, Y, X

    # init_weight = torch.randn(64) # gamma
    # init_bias = torch.randn(64) # beta

    # bn3 = nn.BatchNorm3d(64)
    # print(bn3.weight.data)
    # print(bn3.bias.data)
    # bn3.weight.data = init_weight
    # bn3.bias.data = init_bias
    # print(bn3.weight.data)
    # print(bn3.bias.data)

    # bn1 = nn.BatchNorm1d(64)
    # bn1.weight.data = init_weight
    # bn1.bias.data = init_bias
    # print(bn1.weight.data)
    # print(bn1.bias.data)

    # tensor_size = temp_tensor.size()
    # n_t = tensor_size[0]

    # # from bn3
    # out3 = temp_tensor.permute(1, 2, 0, 3, 4, 5).contiguous()
    # out3 = out3.view(2, 64, 80, 20, 30)
    # out3 = bn3(out3)
    # out3 = out3.view(2, 64, 8, 10, 20, 30)
    # out3 = out3.permute(2, 0, 1, 3, 4, 5).contiguous()
    
    # tensor_size = temp_tensor.size()
    # len_dim = len(tensor_size)
    # changed_idx = list(range(1, len_dim)) + [0]
    # print(changed_idx)
    # print(temp_tensor.shape)
    # out1 = temp_tensor.permute(*changed_idx).contiguous()
    # ret_size = out1.size()
    # print(out1.shape)
    # out1 = out1.view(tensor_size[1], tensor_size[2], -1)
    # out1 = bn1(out1)
    # out1 = out1.view(*ret_size)
    # print(out1.shape)
    # out1 = out1.permute(*([len_dim - 1] + list(range(len_dim - 1)))).contiguous()
    # print(out1.shape)
    # print(out3.shape)

    # print(out3[0,1,:,2,0])
    # print(out1[0,1,:,2,0])
    # print(out3==out1)
    ### Check batch norm equivalency ###

    path_cfg = './configs/cfg_Spiking_RTNH.yml'
    cfg = cfg_from_yaml_file(path_cfg, cfg)

    rdr_sparse = torch.randn(1000, 4) # N, 4
    batch_indices = torch.randint(0, 2, (1000,))
    dict_item = dict(
        rdr_sparse=rdr_sparse.cuda(),
        batch_indices_rdr_sparse=batch_indices.cuda(),
        batch_size=2,
        gt_boxes=torch.randn(2, 10, 7).cuda()
    )
    
    # DenseRTNH
    # dense_rtnh = RtnhDenseBackbone(cfg)
    # dense_rtnh.eval()
    # dict_item = dense_rtnh(dict_item)

    # SpikingRTNH
    spiking_rtnh = SpikingRTNH(cfg).cuda()
    # spiking_rtnh.set_spike_state(True)
    dict_item = spiking_rtnh(dict_item)
    