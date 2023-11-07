"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.12.28
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: utils for common
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import time

from models.skeletons import build_skeleton
import datasets
from configs.config_general import IS_UBUNTU
import configs.config_general as cnf
from utils.util_geometry import *

__all__ = [ 'build_network', \
            'build_optimizer', \
            'build_dataset', \
            'build_scheduler', \
            'vis_tesseract_pline', \
            'set_random_seed', \
            'vis_tesseract_ra_bbox_pline', \
            'get_local_time_str', \
            'dict_datum_to_kitti', \
            'read_imageset_file', \
            'update_dict_feat_not_inferenced', \
            ]

def build_network(p_pline):
    return build_skeleton(p_pline.cfg)

def build_optimizer(p_pline, model):
    lr = p_pline.cfg.OPTIMIZER.LR
    betas = p_pline.cfg.OPTIMIZER.BETAS
    weight_decay = p_pline.cfg.OPTIMIZER.WEIGHT_DECAY
    momentum = p_pline.cfg.OPTIMIZER.MOMENTUM
    
    params = model.parameters()
    if p_pline.cfg.OPTIMIZER.NAME == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif p_pline.cfg.OPTIMIZER.NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif p_pline.cfg.OPTIMIZER.NAME == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    return optimizer

def build_dataset(p_pline, split='train'):
    return datasets.__all__[p_pline.cfg.DATASET.NAME](cfg = p_pline.cfg, split=split)

def build_scheduler(p_pline, optimizer):
    max_epoch = p_pline.cfg.OPTIMIZER.MAX_EPOCH
    batch_size = p_pline.cfg.OPTIMIZER.BATCH_SIZE
    type_total_iter = p_pline.cfg.OPTIMIZER.TYPE_TOTAL_ITER
    try:
        min_lr = p_pline.cfg.OPTIMIZER.MIN_LR
    except:
        print('* Exception error (util_pipeline): No Min LR in Config')
        min_lr = 0

    # print(f'* debug (util_pipeline): dataset num = {p_pline.cfg.DATASET.NUM}')
    if type_total_iter == 'every':
        total_iter = p_pline.cfg.DATASET.NUM // batch_size
    elif type_total_iter == 'all':
        total_iter = (p_pline.cfg.DATASET.NUM // batch_size) * max_epoch
    else:
        print('* Exception error (util_pipeline): No Min LR in Config')

    if p_pline.cfg.OPTIMIZER.SCHEDULER is None:
        return None
    elif p_pline.cfg.OPTIMIZER.SCHEDULER == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min = min_lr)

def set_random_seed(seed, is_cuda_seed=False, is_deterministic=True):
    # We haven't considered cuda seed for released models. The reproduced result could be slightly different.
    # is_deterministic affects speed of training. We only recommend this for getting the final result.
    
    ### b4 ###
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # if is_cuda_seed:
    #     torch.cuda.manual_seed_all(seed)

    # if is_deterministic:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    ### b4 ###

    ### changed ###
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # CUDNN determinism
    torch.backends.cudnn.benchmark = False  # set False whenever input size changes
    np.random.seed(seed)
    random.seed(seed)
    ### changed ###

def vis_tesseract_pline(p_pline, idx=0, vis_type='ra', is_in_deg=True, is_vis_local_maxima_along_range=False):
    '''
    * args
    *   idx: index of data
    *   vis_type: 'ra', 're', 'ae', 'all'
    '''
    datum = p_pline.dataset[idx]
    
    # mean doppler
    tesseract = datum['tesseract'].copy()
    tes_rae = np.mean(tesseract, axis=0)

    # visualize in color heatmap with 2D bbox (XY)
    tes_ra = np.mean(tes_rae, axis=2)
    tes_re = np.mean(tes_rae, axis=1)
    tes_ae = np.mean(tes_rae, axis=0)

    arr_range = p_pline.dataset.arr_range
    arr_azimuth = p_pline.dataset.arr_azimuth
    arr_elevation = p_pline.dataset.arr_elevation

    # print(tes_ra.shape)
    # print(tes_re.shape)
    # print(tes_ae.shape)
    # print(arr_range.shape)
    # print(arr_azimuth.shape)
    # print(arr_elevation.shape)
    
    ### Visualization ###
    if is_in_deg:
        arr_azimuth = arr_azimuth*180./np.pi
        arr_elevation = arr_elevation*180./np.pi

    if not IS_UBUNTU:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if vis_type == 'ra':
        arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')

        plt.colorbar()
        plt.show()
    elif vis_type == 're':
        arr_0, arr_1 = np.meshgrid(arr_elevation, arr_range)

        tes_re_log_scale = 10*np.log10(tes_re)
        if is_vis_local_maxima_along_range:
            min_tes_re_log_scale = np.min(tes_re_log_scale)
            tes_re_local_maxima = np.ones_like(tes_re_log_scale)*min_tes_re_log_scale
            n_row, _ = tes_re_log_scale.shape
            for j in range(n_row):
                arg_maxima = np.argmax(tes_re_log_scale[j,:])
                tes_re_local_maxima[j, arg_maxima] = tes_re_log_scale[j, arg_maxima]
            plt.pcolormesh(arr_0, arr_1, tes_re_local_maxima, cmap='jet')
        else:
            plt.pcolormesh(arr_0, arr_1, tes_re_log_scale, cmap='jet')
        
        plt.colorbar()
        plt.show()
    elif vis_type == 'ae':
        arr_0, arr_1 = np.meshgrid(arr_elevation, arr_azimuth)
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ae), cmap='jet')
        plt.colorbar()
        plt.show()
    elif vis_type == 'all':
        # fig, ax = plt.subplots(1, 1)
        # ax.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')

        return

def vis_tesseract_ra_bbox_pline(p_pline, idx, roi_x, roi_y, is_with_label=True, is_in_deg=True):
    datum = p_pline.dataset[idx]
    
    # mean doppler
    tesseract = datum['tesseract'].copy()
    tes_rae = np.mean(tesseract, axis=0)
    tes_ra = np.mean(tes_rae, axis=2)

    arr_range = p_pline.dataset.arr_range
    arr_azimuth = p_pline.dataset.arr_azimuth
    if is_in_deg:
        arr_azimuth = arr_azimuth*180./np.pi

    if not IS_UBUNTU:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)

    # 이미지 공백 제거
    # https://frhyme.github.io/python-lib/img_savefig_%EA%B3%B5%EB%B0%B1%EC%A0%9C%EA%B1%B0/
    height, width = np.shape(tes_ra)
    # print(height, width)
    figsize = (1, height/width) if height>=width else (width/height, 1)
    plt.figure(figsize=figsize)
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig('./resources/imgs/img_tes_ra.png', bbox_inces='tight', pad_inches=0, dpi=300)
    
    # match the size of image
    temp_img = cv2.imread('./resources/imgs/img_tes_ra.png')
    temp_row, temp_col, _ = temp_img.shape
    # print(temp_row, temp_col)
    if not (temp_row == height and temp_col == width):
        temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./resources/imgs/img_tes_ra.png', temp_img_new)

    plt.close()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')
    plt.colorbar()
    plt.savefig('./resources/imgs/plot_tes_ra.png', dpi=300)

    # Polar to Cartesian (Should flip image)
    ra = cv2.imread('./resources/imgs/img_tes_ra.png')
    ra = np.flip(ra, axis=0)

    # This code is why we have to flip
    # ra_gray = cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # plt.pcolormesh(arr_0, arr_1, ra_gray.astype(float))
    # plt.show()
    # This code is why we have to flip

    arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra, \
        arr_range, arr_azimuth, roi_x=roi_x, roi_y=roi_y, is_in_deg=True)
    
    if is_with_label:
        label = datum['meta']['labels']
        # print(datum['meta'])
        arr_yx_bbox = draw_labels_in_yx_bgr(arr_yx, arr_y, arr_x, label)

    # flip before show
    arr_yx = arr_yx.transpose((1,0,2))
    arr_yx = np.flip(arr_yx, axis=(0,1))
    
    arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
    arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
    # flip before show

    cv2.imshow('Cartesian', arr_yx)
    cv2.imshow('Cartesian (bbox)', arr_yx_bbox)
    cv2.imshow('Front image', cv2.imread(datum['meta']['path_img']))
    plt.show()

    
def draw_labels_in_yx_bgr(arr_yx_in, arr_y_in, arr_x_in, label_in, is_with_bbox_mask=True):
    arr_yx = arr_yx_in.copy()
    arr_y = arr_y_in.copy()
    arr_x = arr_x_in.copy()
    label = label_in.copy()

    y_m_per_pix = np.mean(arr_y[1:] - arr_y[:-1])
    x_m_per_pix = np.mean(arr_x[1:] - arr_x[:-1])
    # print(y_m_per_pix, x_m_per_pix)

    y_min = np.min(arr_y)
    x_min = np.min(arr_x)

    if is_with_bbox_mask:
        row, col, _ = arr_yx.shape
        arr_yx_mask = np.zeros((row, col), dtype=float) # 1 or 0

    dic_cls_bgr = cnf.DIC_CLS_BGR

    for obj in label:
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj

        # calib is already reflected in dataset
        color = dic_cls_bgr[cls_name]
        
        x_pix = (x-x_min)/x_m_per_pix
        y_pix = (y-y_min)/y_m_per_pix
        # y_pix = (-y-y_min)/y_m_per_pix
        # Why? Row 좌표계!, Theta도 Checking 필요
        
        # y_pix = (y-y_min)/y_m_per_pix
        l_pix = l/x_m_per_pix
        w_pix = w/y_m_per_pix
        
        pts = [ [l_pix/2, w_pix/2],
                [l_pix/2, -w_pix/2],
                [-l_pix/2, -w_pix/2],
                [-l_pix/2, w_pix/2]]
        # front left, front right, back right, back left

        # rotation
        # print(f'th = {theta}')
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        # rotation & translation
        pts = list(map(lambda pt: [ x_pix +cos_th*pt[0]-sin_th*pt[1], \
                                    y_pix +sin_th*pt[0]+cos_th*pt[1] ], pts))
        pt_front = (int(np.around((pts[0][0]+pts[1][0])/2)), int(np.around((pts[0][1]+pts[1][1])/2)))
        
        # to integer and tuple
        pts = list(map(lambda pt: (int(np.around(pt[0])), int(np.around(pt[1]))), pts))

        arr_yx = cv2.line(arr_yx, pts[0], pts[1], color, 1)
        arr_yx = cv2.line(arr_yx, pts[1], pts[2], color, 1)
        arr_yx = cv2.line(arr_yx, pts[2], pts[3], color, 1)
        arr_yx = cv2.line(arr_yx, pts[3], pts[0], color, 1)

        # front and center
        pt_cen = (int(np.around(x_pix)), int(np.around(y_pix)))
        arr_yx = cv2.line(arr_yx, pt_cen, pt_front, color, 1)

        # make center black
        arr_yx = cv2.circle(arr_yx, pt_cen, 1, (0,0,0), thickness=-1)

    return arr_yx

def get_local_time_str():
    now = time.localtime()
    tm_year = f'{now.tm_year}'[2:4]
    tm_mon = f'{now.tm_mon}'.zfill(2)
    tm_mday = f'{now.tm_mday}'.zfill(2)
    tm_mday = f'{now.tm_mday}'.zfill(2)
    tm_hour = f'{now.tm_hour}'.zfill(2)
    tm_min = f'{now.tm_min}'.zfill(2)
    tm_sec = f'{now.tm_sec}'.zfill(2)
    return f'{tm_year}{tm_mon}{tm_mday}_{tm_hour}{tm_min}{tm_sec}'

def dict_datum_to_kitti(p_pline, dict_item):
    '''
    * Assuming batch size as 1
    '''
    list_kitti_pred = []
    list_kitti_gt = []
    dict_val_keyword = p_pline.val_keyword # without empty space for cls name

    # considering gt
    header_gt = '0.00 0 0 50 50 150 150'
    for idx_gt, label in enumerate(dict_item['label'][0]): # Assuming batch size as 1
        cls_name, cls_idx, (xc, yc, zc, rz, xl, yl, zl), _ = label
        xc, yc, zc, rz, xl, yl, zl = np.round(xc, 2), np.round(yc, 2), np.round(zc, 2), np.round(rz, 2), np.round(xl, 2), np.round(yl, 2), np.round(zl, 2),
        cls_val_keyword = dict_val_keyword[cls_name]
        # print(cls_val_keyword)
        box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc) # xcam, ycam, zcam
        box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl) # height(ycaml), width(xcaml), length(zcaml)
        str_rot = str(rz)

        kitti_gt = cls_val_keyword + ' ' + header_gt  + ' ' + box_dim  + ' ' + box_centers + ' ' + str_rot
        list_kitti_gt.append(kitti_gt)
    
    if dict_item['pp_num_bbox'] == 0: # empty prediction (should consider for lpc)
        # KITTI Example: Car -1 -1 -4.2780 668.7884 173.1068 727.8801 198.9699 1.4607 1.7795 4.5159 5.3105 1.4764 43.1853 -4.1569 0.9903
        kitti_dummy = 'dummy -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0'
        list_kitti_pred.append(kitti_dummy)
    else:
        list_pp_cls = dict_item['pp_cls']
        header_pred = '-1 -1 0 50 50 150 150'
        for idx_pred, pred_box in enumerate(dict_item['pp_bbox']):
            score, xc, yc, zc, xl, yl, zl, rot = pred_box
            cls_id = list_pp_cls[idx_pred]
            cls_name = p_pline.dict_cls_id_to_name[cls_id]
            cls_val_keyword = dict_val_keyword[cls_name]
            
            box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc)
            box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl)
            str_rot = str(rot)
            str_score = str(score)
            kitti_pred = cls_val_keyword + ' ' + header_pred  + ' ' + box_dim + ' ' + box_centers + ' ' + str_rot + ' ' + str_score
            list_kitti_pred.append(kitti_pred)

    # pp: post-processing
    dict_desc = dict_item['pp_desc']
    capture_time = dict_desc['capture_time']
    road_type = dict_desc['road_type']
    climate = dict_desc['climate']

    dict_item['kitti_pred'] = list_kitti_pred
    dict_item['kitti_gt'] = list_kitti_gt
    dict_item['kitti_desc'] = f'{capture_time}\n{road_type}\n{climate}'
    
    return dict_item

def read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def update_dict_feat_not_inferenced(dict_item):
    '''
    * Assume batch size 1
    * Mostly sleet or adverse weather cases for lpc (e.g., no measurement in ROI)
    '''
    # empty prediction
    # pp: post-processing
    dict_item['pp_bbox'] = None
    dict_item['pp_cls'] = None
    dict_item['pp_desc'] = dict_item['meta'][0]['desc']
    dict_item['pp_num_bbox'] = 0

    return dict_item
