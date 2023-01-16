'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn
import numpy as np
import nms

from utils.Rotated_IoU.oriented_iou_loss import cal_iou

class LdrPillarsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.roi = self.cfg.DATASET.LPC.ROI

        self.voxel_size = self.cfg.MODEL.VOXEL_ENCODER.VOXEL_SIZE
        self.rate_bev_to_feat_size = self.cfg.MODEL.HEAD.RATE_BEV_IMG_TO_BEV_FEAT
        try:
            self.nms_thr = self.cfg.MODEL.HEAD.NMS_OVERLAP_THRESHOLD
        except:
            print('* Exception error (Head): nms threshold is set as 0.3')
            self.nms_thr = 0.3

        ### Anchors ###
        self.anchor_per_grid = []
        num_anchor_temp = 0

        self.list_anchor_classes = []
        self.list_anchor_matched_thr = []
        self.list_anchor_unmatched_thr = []
        self.list_anchor_targ_idx = []
        self.list_anchor_idx = [] # to slice tensor

        # for inference
        self.list_anc_idx_to_cls_id = [] # except bg
        self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        # self.dict_cls_id_to_name = dict()
        # for k, v in self.dict_cls_name_to_id.items():
        #     if v != -1:
        #         self.dict_cls_id_to_name[v] = k

        num_prior_anchor_idx = 0
        for info_anchor in self.cfg.MODEL.ANCHOR_GENERATOR_CONFIG: # per class
            now_cls_name = info_anchor['class_name']
            self.list_anchor_classes.append(now_cls_name)
            self.list_anchor_matched_thr.append(info_anchor['matched_threshold'])
            self.list_anchor_unmatched_thr.append(info_anchor['unmatched_threshold'])
            
            self.anchor_sizes = info_anchor['anchor_sizes']
            self.anchor_rotations = info_anchor['anchor_rotations']
            self.anchor_bottoms = info_anchor['anchor_bottom_heights']

            self.list_anchor_targ_idx.append(num_prior_anchor_idx)
            num_now_anchor = int(len(self.anchor_sizes)*len(self.anchor_rotations)*len(self.anchor_bottoms))
            num_now_anchor_idx = num_prior_anchor_idx+num_now_anchor
            self.list_anchor_idx.append(num_now_anchor_idx)
            num_prior_anchor_idx = num_now_anchor_idx

            for anchor_size in self.anchor_sizes: # per size
                for anchor_rot in self.anchor_rotations: # per rot
                    for anchor_bottom in self.anchor_bottoms: # per bottom: for predicting zc
                        temp_anchor = [anchor_bottom] + anchor_size + [np.cos(anchor_rot), np.sin(anchor_rot)]
                        num_anchor_temp += 1
                        self.anchor_per_grid.append(temp_anchor) # [bot, xl, yl, zl, cos, sin]
                        self.list_anc_idx_to_cls_id.append(self.dict_cls_name_to_id[now_cls_name])
        self.num_anchor_per_grid = num_anchor_temp
        self.num_class = self.cfg.DATASET.CLASS_INFO.NUM_CLS
        self.num_box_code = len(self.cfg.MODEL.HEAD.BOX_CODE)
        ### Anchors ###

        ### 1x1 conv ###
        input_channels = self.cfg.MODEL.HEAD.DIM
        self.conv_cls = nn.Conv2d(
            input_channels, 1 + self.num_anchor_per_grid, # plus one for background
            kernel_size=1
        )
        self.conv_reg = nn.Conv2d(
            input_channels, self.num_anchor_per_grid*self.num_box_code,
            kernel_size=1
        )
        ### 1x1 conv ###

        ### Loss & Logging ###
        self.bg_weight = cfg.MODEL.HEAD.BG_WEIGHT
        self.categorical_focal_loss = FocalLoss()
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING
        ### Loss & Logging ###

        ### Anchor map ###
        self.anchor_map_for_batch = self.create_anchors().cuda() # no batch
        ### Anchor map ###

    def forward(self, dict_item):
        spatial_features_2d = dict_item['spatial_features_2d']
        # print(spatial_features_2d.shape)

        cls_pred = self.conv_cls(spatial_features_2d) # B x num_anchor+1 x Y x X
        reg_pred = self.conv_reg(spatial_features_2d) # B x num_anchor*num_box_code x Y x X

        dict_item['pred'] = {
            'cls': cls_pred,
            'reg': reg_pred,
        }

        return dict_item

    # V2
    def create_anchors(self):
        '''
        * e.g., 2 anchors (a,b) per class for 3 classes (A,B,C),
        *       anchor order -> (Aa Ab Ba Bb Ca Cc)
        '''
        dtype = torch.float32
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        
        vox_x, vox_y, vox_z = self.voxel_size # [m]

        n_x = int((x_max-x_min)/vox_x*self.rate_bev_to_feat_size)
        n_y = int((y_max-y_min)/vox_y*self.rate_bev_to_feat_size)
        # print(n_x, n_y)
        
        # anchor location = center
        grid_size_x = vox_x/(self.rate_bev_to_feat_size)
        grid_size_y = vox_y/(self.rate_bev_to_feat_size)
        # print(grid_size_x, grid_size_y)
        half_grid_size_x = grid_size_x/2.
        half_grid_size_y = grid_size_y/2.
        anchor_y = torch.arange(y_min, y_max, grid_size_x, dtype=dtype) + half_grid_size_x # plus for ldr
        anchor_x = torch.arange(x_min, x_max, grid_size_y, dtype=dtype) + half_grid_size_y
        # print(anchor_y) # 200
        # print(anchor_x) # 248

        anchor_y = anchor_y.repeat_interleave(n_x)
        anchor_x = anchor_x.repeat(n_y)
        # print(anchor_y.shape) # 49600
        # print(anchor_x.shape) # 49600

        flattened_anchor_map = torch.stack((anchor_x, anchor_y), dim=1).unsqueeze(0).repeat(self.num_anchor_per_grid, 1, 1)
        # print(flattened_anchor_map.shape) # 2 x 49600 x 2 (xc, yc)
        flattened_anchor_attr = torch.tensor(self.anchor_per_grid, dtype=dtype)
        # print(flattened_anchor_attr.shape) # 2 x 6 (bottom, xl, yl, zl, cos, sin)
        flattened_anchor_attr = flattened_anchor_attr.unsqueeze(1).repeat(1, flattened_anchor_map.shape[1], 1)
        # print(flattened_anchor_attr.shape) # 2 x 49600 x 6

        anchor_map = torch.cat((flattened_anchor_map, flattened_anchor_attr), \
            dim=-1).view(self.num_anchor_per_grid, n_y, n_x, 8).contiguous().permute(0,3,1,2)
        anchor_map = anchor_map.reshape(-1, n_y, n_x).contiguous() # 16, 200, 248
        # print(anchor_map.shape) # 2 * (2+6) x 200 x 248

        anchor_map_for_batch = anchor_map.unsqueeze(0) # 1 x 16 x 200 x 248

        return anchor_map_for_batch

    def loss(self, dict_item):
        cls_pred = dict_item['pred']['cls']
        reg_pred = dict_item['pred']['reg']

        dtype, device = cls_pred.dtype, cls_pred.device 
        B, _, n_y, n_x = cls_pred.shape
        num_grid_per_anchor = int(n_y*n_x)
        # print(num_grid_per_anchor)

        anchor_maps = self.anchor_map_for_batch.repeat(B, 1, 1, 1)

        reg_pred = anchor_maps + reg_pred # prediction = residual

        # for iou calculation
        cls_pred = cls_pred.view(B, 1+self.num_anchor_per_grid, n_y, n_x)
        reg_pred = reg_pred.view(B, self.num_anchor_per_grid, -1, n_y, n_x)
        
        # make labels
        anc_idx_targets = torch.full((B, n_y, n_x), -1, dtype = torch.long, device = device)

        pos_reg_pred = []
        pos_reg_targ = []

        is_label_contain_objs = False # at least one
        for batch_idx, list_objs in enumerate(dict_item['label']):
            if len(list_objs) != 0:
                is_label_contain_objs = True

            prior_anc_idx = 0
            list_anchor_per_cls = []
            for idx_anc_cls, anc_cls_name in enumerate(self.list_anchor_classes):
                now_anc_idx = self.list_anchor_idx[idx_anc_cls]
                # x,y,xl,yl,theta (sin, cos)
                temp_anc = torch.cat(\
                        (reg_pred[batch_idx,prior_anc_idx:now_anc_idx,:2],\
                         reg_pred[batch_idx,prior_anc_idx:now_anc_idx,3:5],\
                         torch.atan(reg_pred[batch_idx,prior_anc_idx:now_anc_idx,6:7]/reg_pred[batch_idx,prior_anc_idx:now_anc_idx,5:6])), dim=1)
                # print(temp_anc.shape) # n_anchor x 5 x n_y x n_x
                temp_anc = temp_anc.permute(0, 2, 3, 1).contiguous()
                temp_anc = temp_anc.view(1,-1,5)
                # print(temp_anc.shape) # 1 x (n_anchor*n_y*n_x) x 5
                list_anchor_per_cls.append(temp_anc)
                prior_anc_idx = now_anc_idx
            
            for label_idx, label in enumerate(list_objs):
                cls_name, cls_id, (xc, yc, zc, rz, xl, yl, zl), _ = label

                # find anc idx
                idx_anc_cls = self.list_anchor_classes.index(cls_name)
                pred_anchors = list_anchor_per_cls[idx_anc_cls]
                cls_targ_idx = self.list_anchor_targ_idx[idx_anc_cls]
                matched_iou_thr = self.list_anchor_matched_thr[idx_anc_cls]
                unmatched_iou_thr = self.list_anchor_unmatched_thr[idx_anc_cls]
                # print(cls_targ_idx, matched_iou_thr, unmatched_iou_thr)

                label_anchor = torch.tensor([xc, yc, xl, yl, rz], dtype=dtype, device=device)
                # making same size as pred_anchors
                label_anchor = label_anchor.unsqueeze(0).unsqueeze(0).repeat(1, pred_anchors.shape[1], 1)

                iou, _, _, _ = cal_iou(label_anchor, pred_anchors)

                pos_iou_anc_idx = torch.where(iou > matched_iou_thr)[1]
                # make at least 1 pos box
                if len(pos_iou_anc_idx) == 0:
                    # print(torch.argmax(iou))
                    # print(torch.max(iou))
                    pos_iou_anc_idx = (torch.argmax(iou)).reshape(1)

                neg_iou_anc_idx = torch.where(iou < unmatched_iou_thr)[1]
                # print(torch.max(neg_iou_anc_idx))
                neg_iou_anc_idx = torch.remainder(neg_iou_anc_idx, num_grid_per_anchor)
                # print(torch.max(neg_iou_anc_idx))

                # print(iou.shape) # check total sum = iou
                # print(pos_iou_anc_idx.shape)
                # print(torch.where(torch.logical_and(iou<=matched_iou_thr, iou>=unmatched_iou_thr))[1].shape)
                # print(neg_iou_anc_idx.shape)

                idx_y_neg = torch.div(neg_iou_anc_idx, n_x, rounding_mode='trunc')
                idx_x_neg = torch.remainder(neg_iou_anc_idx, n_x)

                anc_idx_targets[batch_idx, idx_y_neg, idx_x_neg] = 0 # 'Background'

                pos_iou_anc_idx_offset = torch.div(pos_iou_anc_idx, \
                    num_grid_per_anchor, rounding_mode='trunc') # for classification
                pos_iou_anc_idx = torch.remainder(pos_iou_anc_idx, num_grid_per_anchor) # for Y, X

                idx_y_pos = torch.div(pos_iou_anc_idx, n_x, rounding_mode='trunc')
                idx_x_pos = torch.remainder(pos_iou_anc_idx, n_x)
                
                # 1 for background, cls_targ_idx for class, offset for anc_idx
                temp_anc_idx_targets = cls_targ_idx + pos_iou_anc_idx_offset
                anc_idx_targets[batch_idx, idx_y_pos, idx_x_pos] = 1 + temp_anc_idx_targets # plus 1 for background

                temp_reg_box_pred = reg_pred[batch_idx,temp_anc_idx_targets,:,idx_y_pos,idx_x_pos]
                temp_num_pos, _ = temp_reg_box_pred.shape
                temp_reg_box_targ = torch.tensor([[xc, yc, zc, xl, yl, zl, \
                    np.cos(rz), np.sin(rz)]], dtype = dtype, device = device).repeat((temp_num_pos, 1))
                # print(temp_reg_box_targ.shape)

                pos_reg_pred.append(temp_reg_box_pred) # remain batch dim for concat
                pos_reg_targ.append(temp_reg_box_targ)
        
        if not is_label_contain_objs: # All batches without objs
            loss_reg = 0.
            loss_cls = 0. # focal loss
        else:
            ### Focal loss ###
            counted_anc_idx = torch.where(anc_idx_targets > -1) # pos and neg boxes only

            # anc_idx (long)
            anc_idx_targets_counted = anc_idx_targets[counted_anc_idx]
            # print(anc_idx_targets_counted.shape)

            # logit
            anc_logit_counted = cls_pred[counted_anc_idx[0],:,counted_anc_idx[1],counted_anc_idx[2]]
            # print(anc_logit_counted.shape)

            # weight
            anc_cls_weights = torch.ones(1+self.num_anchor_per_grid, device = device)
            for idx_anc in range(1+self.num_anchor_per_grid):
                len_targ_anc = float(len(torch.where(anc_idx_targets_counted==idx_anc)[0]))
                if idx_anc == 0: # background
                    temp_weight = self.bg_weight/len_targ_anc
                else:
                    if len_targ_anc == 0.: # if nothing in such class
                        temp_weight = 0.
                    else:
                        temp_weight = 1./len_targ_anc
                anc_cls_weights[idx_anc] = min(temp_weight, 1.)

            self.categorical_focal_loss.weight = anc_cls_weights
            loss_cls = self.categorical_focal_loss(anc_logit_counted, anc_idx_targets_counted)
            ### Focal loss ###

            pos_reg_pred = torch.cat(pos_reg_pred, dim=0)
            pos_reg_targ = torch.cat(pos_reg_targ, dim=0)
            loss_reg = torch.nn.functional.smooth_l1_loss(pos_reg_pred, pos_reg_targ)
        total_loss = loss_cls + loss_reg

        if self.is_logging:
            dict_item['logging'] = dict()
            dict_item['logging'].update(self.logging_dict_loss(total_loss, 'total_loss'))
            dict_item['logging'].update(self.logging_dict_loss(loss_reg, 'loss_reg'))
            dict_item['logging'].update(self.logging_dict_loss(loss_cls, 'focal_loss_cls'))

        return total_loss

    def logging_dict_loss(self, loss, name_key):
        try:
            log_loss = loss.cpu().detach().item()
        except:
            log_loss = loss # for 0. loss

        return {name_key: log_loss}

    ### Validation & Inference ###
    def get_nms_pred_boxes_for_single_sample(self, dict_item, conf_thr, is_nms=True):
        '''
        * This function is common function of head for validataion & inference
        * For convenience, we assume batch_size = 1
        '''
        cls_pred = dict_item['pred']['cls'][0] # (1+n_anc) x n_y x n_x
        reg_pred = dict_item['pred']['reg'][0] # n_anc x n_y x n_x
        anchor_map = self.anchor_map_for_batch[0]
        reg_pred = anchor_map + reg_pred
        
        device = cls_pred.device

        # n_y x n_x -> (n_y*n_x)
        cls_pred = cls_pred.view(cls_pred.shape[0], -1)
        reg_pred = reg_pred.view(reg_pred.shape[0], -1)

        # bg X & more than conf_thr
        cls_pred = torch.softmax(cls_pred, dim=0)
        idx_deal = torch.where(
            (torch.argmax(cls_pred, dim=0)!=0) & (torch.max(cls_pred, dim=0)[0]>conf_thr))

        # for finding cls (not anc idx)
        tensor_anc_idx_per_cls = torch.tensor(self.list_anc_idx_to_cls_id, dtype=torch.long, device=device)
        
        len_deal_anc = len(idx_deal[0])
        # print('* debug # of dealing anchor boxes == n_anc:', len_deal_anc)
        if len_deal_anc > 0: # for only dealing grids (not bg & more than conf)
            grid_anc_cls_logit = cls_pred[:, idx_deal[0]] # logit x n_pred / slice 1 for bg
            grid_anc_cls_idx = torch.argmax(grid_anc_cls_logit, dim=0) # minus 1 for bg after get conf
            grid_reg = reg_pred[:, idx_deal[0]]

            # to arange anc
            idx_range_anc = torch.arange(0, len_deal_anc, dtype=torch.long, device=device)
            anc_conf_score = grid_anc_cls_logit[grid_anc_cls_idx,idx_range_anc].unsqueeze(0) # 1 x n_deal
            grid_anc_cls_idx = grid_anc_cls_idx -1 # minus 1 for bg
            # print(anc_conf_score) # check if it is larger than conf_thr

            list_sliced_reg_bbox = []
            idx_slice_start = (grid_anc_cls_idx*self.num_box_code).long()
            
            for idx_reg_value in range(self.num_box_code):
                list_sliced_reg_bbox.append(grid_reg[idx_slice_start+idx_reg_value,idx_range_anc]) # n_deal
            sliced_reg_bbox = torch.stack(list_sliced_reg_bbox)
            # print(sliced_reg_bbox.shape) # 8 x n_deal
            temp_angle = torch.atan2(sliced_reg_bbox[-1,:], sliced_reg_bbox[-2,:]).unsqueeze(0)
            # print(temp_angle.shape)
            pred_reg_bbox_with_conf = torch.cat((anc_conf_score, sliced_reg_bbox[:-2,:], temp_angle), dim=0) # conf, x, y, z, xl, yl, zl, theta
            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.transpose(0,1)
            # print(pred_reg_bbox_with_conf.shape) # n_anc x 8 (score, x, y, z, xl, yl, zl, theta)

            cls_id_per_anc = tensor_anc_idx_per_cls[grid_anc_cls_idx]
            # print(cls_id_per_anc.shape) # n_anc
            num_of_bbox = len_deal_anc
            # print('* debug before nms: ', num_of_bbox)

            ### nms ###
            try:
                if is_nms:
                    pred_reg_xy_xlyl_th = torch.cat((pred_reg_bbox_with_conf[:,1:3], \
                    pred_reg_bbox_with_conf[:,4:6], pred_reg_bbox_with_conf[:,7:8]), dim=1).cpu().detach().numpy()
                
                    c_list = list(map(tuple, pred_reg_xy_xlyl_th[:,:2]))

                    ### Assert error (Height > 0) ###
                    dim_list = list(map(np.abs, pred_reg_xy_xlyl_th[:,2:4]))
                    ### Assert error (Height > 0) ###
                    
                    dim_list = list(map(tuple, pred_reg_xy_xlyl_th[:,2:4]))
                    angle_list = list(map(float, pred_reg_xy_xlyl_th[:,4]))

                    list_tuple_for_nms = [[a, b, c] for (a, b, c) in zip(c_list, dim_list, angle_list)]
                    conf_score = pred_reg_bbox_with_conf[:, 0:1].cpu().detach().numpy()
                
                    indices = nms.rboxes(list_tuple_for_nms, conf_score, nms_threshold=self.nms_thr)
                    pred_reg_bbox_with_conf = pred_reg_bbox_with_conf[indices]
                    cls_id_per_anc = cls_id_per_anc[indices]

                    num_of_bbox = len(indices) # after nms
            except:
                print('* Exception error (head.py): nms error, probably assert height > 0')

            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.cpu().detach().numpy().tolist()
            cls_id_per_anc = cls_id_per_anc.cpu().detach().numpy().tolist()
            # print('* debug after nms: ', num_of_bbox)
        else:
            # empty prediction
            pred_reg_bbox_with_conf = None
            cls_id_per_anc = None
            num_of_bbox = 0 # 0

        # pp: post-processing
        dict_item['pp_bbox'] = pred_reg_bbox_with_conf # score, x, y, z, xl, yl, zl, theta
        dict_item['pp_cls'] = cls_id_per_anc
        dict_item['pp_desc'] = dict_item['meta'][0]['desc']
        dict_item['pp_num_bbox'] = num_of_bbox
        # print(dict_item['label'][0])
        # print(dict_item['pp_bbox'])

        return dict_item
    ### Validation & Inference ###

class FocalLoss(nn.Module):
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return nn.functional.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
