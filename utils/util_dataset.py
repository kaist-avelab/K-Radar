"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2022.05.29
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: utils for common
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path as osp
import open3d as o3d
import pickle
import os

from utils.util_geometry import *
from utils.util_geometry import Object3D

__all__ = [ 'func_show_radar_tensor_bev', \
            'func_show_lidar_point_cloud', \
            'func_show_gaussian_confidence_cart', \
            'func_show_gaussian_confidence_polar_color', \
            'func_show_gaussian_confidence_polar', \
            'func_show_heatmap_polar_with_bbox', \
            'func_generate_gaussian_conf_labels', \
            'func_show_radar_cube_bev', \
            'func_show_sliced_radar_cube', \
            'func_show_rdr_pc_cube', \
            'func_show_rdr_pc_tesseract', \
            'func_save_undistorted_camera_imgs_w_projected_params', \
            'func_get_distribution_of_label', \
            ]

def func_show_radar_tensor_bev(p_pline, dict_item, bboxes=None, \
        roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_return_bbox_bev_tensor=False, alpha=0.9, lthick=1, infer=None, infer_gt=None, norm_img=None):
    rdr_tensor = p_pline.get_tesseract(dict_item['meta']['path_rdr_tesseract'])
    rdr_bev = np.mean(np.mean(rdr_tensor, axis=0), axis=2)

    arr_range = p_pline.arr_range
    arr_azimuth = p_pline.arr_azimuth
    
    arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)

    height, width = np.shape(rdr_bev)
    # print(height, width)
    # figsize = (1, height/width) if height>=width else (width/height, 1)
    # plt.figure(figsize=figsize)
    plt.clf()
    plt.cla()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig('./resources/imgs/img_tes_ra.png', bbox_inches='tight', pad_inches=0, dpi=300)
    
    temp_img = cv2.imread('./resources/imgs/img_tes_ra.png')
    temp_row, temp_col, _ = temp_img.shape
    if not (temp_row == height and temp_col == width):
        temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./resources/imgs/img_tes_ra.png', temp_img_new)

    plt.close()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
    plt.colorbar()
    plt.savefig('./resources/imgs/plot_tes_ra.png', dpi=300)

    # Polar to Cartesian (Should flip image)
    ra = cv2.imread('./resources/imgs/img_tes_ra.png')
    ra = np.flip(ra, axis=0)

    arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra, arr_range, arr_azimuth, \
        roi_x = roi_x, roi_y = roi_y, is_in_deg=False)

    ### Image processing ###
    if norm_img is None:
        pass
    elif norm_img.split('_')[0] == 'hist': # histogram stretching
        arr_yx = cv2.normalize(arr_yx, None, 0, 255, cv2.NORM_MINMAX)
    elif norm_img.split('_')[0] == 'alp':
        alp = float(norm_img.split('_')[1])
        arr_yx = np.clip((1+alp)*arr_yx - 128*alp, 0, 255).astype(np.uint8)

    if not (bboxes is None):
        ### Original ###
        # arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx, arr_y, arr_x, bboxes, lthick=lthick)
        # if is_return_bbox_bev_tensor:
        #     return arr_yx_bbox
        arr_yx_bbox = arr_yx.copy()

        ### inference ###
        if infer_gt is not None:
            bboxes_gt = []
            for idx_obj, obj in enumerate(infer_gt):
                bboxes_gt.append(['Gt Sedan', 0, [obj.xc, obj.yc, obj.zc, obj.rot_rad, obj.xl, obj.yl, obj.zl], idx_obj])
            arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx_bbox, arr_y, arr_x, bboxes_gt, lthick=lthick)
        
        if infer is not None:
            # print(infer)
            bboxes_infer = []
            for idx_obj, obj in enumerate(infer):
                bboxes_infer.append(['Infer', 0, [obj.xc, obj.yc, obj.zc, obj.rot_rad, obj.xl, obj.yl, obj.zl], idx_obj])
            arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx_bbox, arr_y, arr_x, bboxes_infer, lthick=lthick)

    # alpha = 0.9
    arr_yx_bbox = cv2.addWeighted(arr_yx_bbox, alpha, arr_yx, 1 - alpha, 0)

    # flip before show
    # arr_yx = arr_yx.transpose((1,0,2))
    # arr_yx = np.flip(arr_yx, axis=(0,1))

    # print(dict_item['meta'])
    # cv2.imshow('Cartesian', arr_yx)

    if not (bboxes is None):
        arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
        arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
        cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx_bbox,(0,0),fx=4,fy=4))
    else:
        arr_yx = arr_yx.transpose((1,0,2))
        arr_yx = np.flip(arr_yx_bbox, axis=(0,1))
        cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx,(0,0),fx=2,fy=2))

    # cv2.imshow('Front image', cv2.imread(dict_item['meta']['path_cam_front'])[:,:1280,:])
    # plt.show()
    cv2.waitKey(0)

def func_show_lidar_point_cloud(p_pline, dict_item, bboxes=None, \
        roi_x=[0, 100], roi_y=[-50, 50], roi_z=[-10, 10]):
    pc_lidar = dict_item['ldr_pc_64']
    # ROI filtering
    pc_lidar = pc_lidar[
        np.where(
            (pc_lidar[:, 0] > roi_x[0]) & (pc_lidar[:, 0] < roi_x[1]) &
            (pc_lidar[:, 1] > roi_y[0]) & (pc_lidar[:, 1] < roi_y[1]) &
            (pc_lidar[:, 2] > roi_z[0]) & (pc_lidar[:, 2] < roi_z[1])
        )
    ]

    bboxes_o3d = []
    for obj in bboxes:
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
        # try item()
        bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta))

        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                [4, 5], [6, 7], #[5, 6],[4, 7],
                [0, 4], [1, 5], [2, 6], [3, 7],
                [0, 2], [1, 3], [4, 6], [5, 7]]
        colors_bbox = [p_pline.cfg.VIS.CLASS_RGB[cls_name] for _ in range(len(lines))]

    line_sets_bbox = []
    for gt_obj in bboxes_o3d:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors_bbox)
        line_sets_bbox.append(line_set)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # Display the bounding boxes:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])

    o3d.visualization.draw_geometries([pcd] + line_sets_bbox)

def func_show_gaussian_confidence_cart(p_pline, roi_x=[0, 0.2, 100], roi_y=[-80, 0.2, 80], bboxes=None):
    arr_yx_conf, arr_y, arr_x = get_gaussian_confidence_cart(roi_x=roi_x, roi_y=roi_y, \
                                                    bboxes=bboxes, is_vis=False, is_for_bbox_vis=True)

    if not (bboxes is None):
        arr_yx_conf = cv2.cvtColor((arr_yx_conf*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        arr_yx_conf = draw_bbox_in_yx_bgr(arr_yx_conf, arr_y, arr_x, bboxes)
        arr_yx_conf = arr_yx_conf.transpose((1,0,2))
        arr_yx_conf = np.flip(arr_yx_conf, axis=(0,1))
    else:
        # Flip b4 show (Cartesian)
        arr_yx_conf = arr_yx_conf.transpose((1,0))
        arr_yx_conf = np.flip(arr_yx_conf, axis=(0,1))

    cv2.imshow('Conf (Cart)', arr_yx_conf)
    cv2.waitKey(0)

def func_show_gaussian_confidence_polar_color(p_pline, arr_range=None, arr_azimuth=None, \
                                    roi_x=[0, 0.2, 100], roi_y=[-80, 0.2, 80], bboxes=None):
    if arr_range is None:
        arr_range = p_pline.arr_range
    
    if arr_azimuth is None:
        arr_azimuth = p_pline.arr_azimuth

    arr_yx_conf, arr_y, arr_x = get_gaussian_confidence_cart(roi_x=roi_x, roi_y=roi_y, \
                                                bboxes=bboxes, is_vis=False, is_for_bbox_vis=True)
    arr_yx_conf = cv2.cvtColor((arr_yx_conf*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    arr_yx_conf = draw_bbox_in_yx_bgr(arr_yx_conf, arr_y, arr_x, bboxes)
    arr_ra_conf = change_arr_cart_to_polar_2d(arr_yx_conf, roi_x, roi_y, arr_range, arr_azimuth, dtype='color')
    
    for bbox in bboxes:
        _, idx_cls, reg_params, _ = bbox
        x, y, _, theta, l, w, _ = reg_params

        r = np.sqrt(x**2+y**2)
        azi = np.arctan2(-y, x)

        idx_r = np.argmin(np.abs(arr_range-r))
        idx_a = np.argmin(np.abs(arr_azimuth-azi))

        cv2.circle(arr_ra_conf, (idx_a,idx_r), 1, (0,255,0), thickness=-1)
    
    cv2.imshow('Conf (Polar)', arr_ra_conf)
    cv2.waitKey(0)

def func_show_gaussian_confidence_polar(p_pline, arr_range=None, arr_azimuth=None, \
                                roi_x=[0, 0.2, 100], roi_y=[-80, 0.2, 80], bboxes=None):
    if arr_range is None:
        arr_range = p_pline.arr_range
    
    if arr_azimuth is None:
        arr_azimuth = p_pline.arr_azimuth

    arr_yx_conf = get_gaussian_confidence_cart(roi_x=roi_x, roi_y=roi_y, bboxes=bboxes)
    arr_ra_conf = change_arr_cart_to_polar_2d(arr_yx_conf, roi_x, roi_y, arr_range, arr_azimuth)

    ### Debug ###
    arr_ra_conf = cv2.cvtColor((arr_ra_conf*255.).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    print(f'shape = {arr_ra_conf.shape}')
    print(bboxes)
    for bbox in bboxes:
        _, idx_cls, reg_params, _ = bbox
        x, y, _, theta, l, w, _ = reg_params

        print(x, y)

        r = np.sqrt(x**2+y**2)
        azi = np.arctan2(-y, x)

        idx_r = np.argmin(np.abs(arr_range-r))
        idx_a = np.argmin(np.abs(arr_azimuth-azi))
        print('>r:', r, arr_range[idx_r-1], arr_range[idx_r], arr_range[idx_r+1])
        print('>a:', azi, arr_azimuth[idx_a-1], arr_azimuth[idx_a], arr_azimuth[idx_a+1])
        

        cv2.circle(arr_ra_conf, (idx_a,idx_r), 1, (0,0,255), thickness=-1)
    
    ### Debug ###
    # Flip b4 show (Polar)
    # arr_ra_conf = np.flip(arr_ra_conf, axis=0)
    ### Debug ###

    cv2.imshow('Conf (Polar)', arr_ra_conf)
    cv2.waitKey(0)

def func_show_heatmap_polar_with_bbox(p_pline, idx_datum, scale):
    print('Takes quiet a lot to transform due to inverse transform')
    label = p_pline.get_label_bboxes(p_pline[idx_datum]['meta']['path_label'], p_pline[idx_datum]['calib_info'])
    arr_yx_bbox = p_pline.show_radar_tensor_bev(p_pline[idx_datum], bboxes=label, \
                roi_x=[0,0.2,120], roi_y=[-100,0.2,100], is_return_bbox_bev_tensor=True)

    arr_r = p_pline.arr_range
    arr_a = p_pline.arr_azimuth

    arr_r = get_high_resolution_array(arr_r, scale)
    arr_a = get_high_resolution_array(arr_a, scale)

    arr_ra_bbox = change_arr_cart_to_polar_2d(arr_yx_bbox, roi_x=[0,0.2,120], roi_y=[-100,0.2,100], \
                            arr_range=arr_r, arr_azimuth=arr_a, dtype='color')

    # Flip b4 show (Cartesian)
    arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
    arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
    
    # Flip b4 show (Polar)
    arr_ra_bbox = np.flip(arr_ra_bbox, axis=0)

    cv2.imshow('Bboxes (Polar)', arr_ra_bbox)
    cv2.imshow('Bboxes (Cart)', arr_yx_bbox)
    cv2.waitKey(0)

def func_generate_gaussian_conf_labels(p_pline, dir_gen, gen_type='polar', \
            roi_x_res=[0.00, 0.16, 69.12], roi_y_res=[-39.68, 0.16, 39.68]):
    '''
    * This is function for generating pre-defined confidence labels
    * gen_type in ['polar', 'cart']
    * setting the resolution of generated labels by adjusting the roi_o_res
    '''
    from tqdm import tqdm

    # Setting the resolution of cartesian coord for inverse transform
    roi_x = roi_x_res
    roi_y = roi_y_res

    # Create a label directory
    label_dir = osp.join(dir_gen, p_pline.cfg.GENERAL.NAME)
    os.makedirs(label_dir, exist_ok=True)

    for idx_datum in tqdm(range(len(p_pline))):
        path_label = p_pline[idx_datum]['meta']['path_label']
        bboxes = p_pline.get_label_bboxes(path_label)
        
        arr_yx_conf = get_gaussian_confidence_cart(roi_x=roi_x, roi_y=roi_y, bboxes=bboxes)
        
        if gen_type == 'polar':
            arr_ra_conf = change_arr_cart_to_polar_2d(arr_yx_conf, roi_x, roi_y, p_pline.arr_range, p_pline.arr_azimuth)
        
        ### Visualization ###
        # print(bboxes)
        # cv2.imshow('arr_ra_conf', arr_ra_conf)
        # cv2.waitKey(10)
        ### Visualization ###

        dir_gen = osp.join(label_dir, path_label.split('/')[-3])
        os.makedirs(dir_gen, exist_ok=True)
        file_name = path_label.split('/')[-1].split('.')[0]
        path_gen = osp.join(dir_gen, f'{file_name}.bin')
        with open(path_gen, 'wb') as f:
            if gen_type == 'polar':
                pickle.dump(arr_ra_conf, f)
            elif gen_type == 'cart':
                pickle.dump(arr_yx_conf, f)
            else:
                raise AttributeError('polar or cart')

def func_show_radar_cube_bev(p_pline, dict_item, bboxes=None, magnifying=4, is_with_doppler = False, is_with_log = False):
    rdr_cube, rdr_cube_mask, rdr_cube_cnt = p_pline.get_cube(dict_item['meta']['path']['rdr_cube'], mode=0)
    if is_with_doppler:
        rdr_cube_doppler = p_pline.get_cube_doppler(dict_item['meta']['path']['rdr_cube_doppler'])
    print(bboxes)
    # print(np.unique(rdr_cube)) # check -1
    # if -1. in np.unique(rdr_cube):
    #     print('-1 is in cube')

    # print(rdr_cube.shape)
    # print(rdr_cube_cnt.shape)
    # print(np.unique(p_pline.arr_bev_none_minus_1))
    # cv2.imshow('hi', p_pline.arr_bev_none_minus_1.astype(np.uint8))
    # cv2.waitKey(0)

    rdr_cube = np.sum(rdr_cube, axis=0)
    # print(np.max(rdr_cube))
    # print(np.min(rdr_cube))
    rdr_cube = rdr_cube/rdr_cube_cnt
    # print(np.max(rdr_cube))
    # print(np.min(rdr_cube))

    # log
    if is_with_log:
        rdr_cube = np.maximum(rdr_cube, 1.)
        rdr_cube_bev = 10*np.log10(rdr_cube)
    else:
        rdr_cube_bev = rdr_cube

    if is_with_doppler:
        rdr_cube_doppler = np.sum(rdr_cube_doppler, axis=0)
        rdr_cube_doppler = rdr_cube_doppler/rdr_cube_cnt

    normalizing = 'min_max'
    if normalizing == 'max':
        rdr_cube_bev = rdr_cube_bev/np.max(rdr_cube_bev)
    elif normalizing == 'fixed':
        rdr_cube_bev = rdr_cube_bev/p_pline.cfg.DATASET.RDR_CUBE.NORMALIZING.VALUE
    elif normalizing == 'min_max':
        ### Care for Zero parts ###
        min_val = np.min(rdr_cube_bev[rdr_cube_bev!=0.])
        max_val = np.max(rdr_cube_bev[rdr_cube_bev!=0.])
        rdr_cube_bev[rdr_cube_bev!=0.]= (rdr_cube_bev[rdr_cube_bev!=0.]-min_val)/(max_val-min_val)

        # rdr_cube_bev = (rdr_cube_bev-np.min(rdr_cube_bev))/(np.max(rdr_cube_bev)-np.min(rdr_cube_bev))
    rdr_cube_bev = rdr_cube_bev
    # print(np.max(rdr_cube_bev))
    # print(np.min(rdr_cube_bev))

    arr_0, arr_1 = np.meshgrid(p_pline.arr_x_cb, p_pline.arr_y_cb)
    # print(p_pline.arr_y_cb.shape)
    # print(p_pline.arr_x_cb.shape)
    # print(p_pline.list_roi_idx_cb)
    # print(p_pline.arr_x_cb[0], p_pline.arr_x_cb[-1])
    # print(p_pline.arr_y_cb[0], p_pline.arr_y_cb[-1])
    # print(np.max(rdr_cube_bev))
    # print(np.min(rdr_cube_bev))

    rdr_cube_bev_vis = cv2.cvtColor((rdr_cube_bev*255.).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for bbox in bboxes:
        _, _, [x, y, z, theta, xl, yl, zl], _ = bbox
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)

        idx_x = np.argmin(np.abs(p_pline.arr_x_cb-x))
        idx_y = np.argmin(np.abs(p_pline.arr_y_cb-y))
        rdr_cube_bev_vis = cv2.circle(rdr_cube_bev_vis, (idx_x,idx_y),1,(0,0,255),thickness=-1) # center

        # print(obj3d.corners.shape)
        pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
        
        for pt in pts:
            idx_x = np.argmin(np.abs(p_pline.arr_x_cb-pt[0]))
            idx_y = np.argmin(np.abs(p_pline.arr_y_cb-pt[1]))
            rdr_cube_bev_vis = cv2.circle(rdr_cube_bev_vis, (idx_x,idx_y),1,(255,0,0),thickness=-1)

    cv2.imshow('jet map', cv2.resize(rdr_cube_bev_vis, (0,0),fx=4,fy=4))

    ### Jet map visualization ###
    rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
    height, width = np.shape(rdr_cube_bev)
    # print(height, width)

    # print(rdr_cube_bev.shape)
    figsize = (1*magnifying, height/width*magnifying) if height>=width else (width/height*magnifying, 1*magnifying)
    plt.figure(figsize=figsize)
    plt.pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
    plt.colorbar()
    plt.title('power')
    if is_with_doppler:
        plt.figure(figsize=figsize)
        plt.pcolormesh(arr_0, arr_1, rdr_cube_doppler, cmap=plt.colormaps['PiYG'])
        plt.colorbar()
        plt.title('doppler')
    plt.show()

def func_show_sliced_radar_cube(p_pline, dict_item, bboxes=None, magnifying=4, idx_custom_slice=None):
    rdr_cube, rdr_cube_cnt = p_pline.get_cube(dict_item['meta']['path_rdr_cube'], mode=0)
    
    ### Infomation for pre-processing ###
    print(bboxes)
    arr_x, arr_y, arr_z = p_pline.arr_x_cb, p_pline.arr_y_cb, p_pline.arr_z_cb
    num_z, num_y, num_x = rdr_cube.shape
    print(num_z, num_y, num_x)
    ### Infomation for pre-processing ###
    
    ### Change here like show radar cube function! ###
    ### Pre-processing ###
    rdr_cube_bev = np.sum(rdr_cube, axis=0)
    rdr_cube_bev = rdr_cube_bev/rdr_cube_cnt
    rdr_cube_bev = np.maximum(rdr_cube_bev, 1.)
    rdr_cube_bev = 10*np.log10(rdr_cube_bev)
    rdr_cube_bev = rdr_cube_bev/np.max(rdr_cube_bev)
    if bboxes is None:
        bboxes = []
    ### Pre-processing ###

    ### Vis for Opencv ###
    rdr_cube_bev_vis = cv2.cvtColor((rdr_cube_bev*255.).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for bbox in bboxes:
        _, _, [x, y, z, theta, xl, yl, zl], _ = bbox
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)
        idx_x = np.argmin(np.abs(p_pline.arr_x_cb-x))
        idx_y = np.argmin(np.abs(p_pline.arr_y_cb-y))
        rdr_cube_bev_vis = cv2.circle(rdr_cube_bev_vis, (idx_x,idx_y),1,(0,0,255),thickness=-1) # center
        pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
        for idx_pt, pt in enumerate(pts):
            idx_x = np.argmin(np.abs(p_pline.arr_x_cb-pt[0]))
            idx_y = np.argmin(np.abs(p_pline.arr_y_cb-pt[1]))
            rdr_cube_bev_vis = cv2.circle(rdr_cube_bev_vis, (idx_x,idx_y),1,(255,0,0),thickness=-1)
            cv2.putText(rdr_cube_bev_vis, f'{idx_pt}', (idx_x,idx_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0,255,255))
    rdr_cube_bev_vis = cv2.resize(rdr_cube_bev_vis, (0,0), fx=magnifying,fy=magnifying)
    ### Vis for Opencv ###

    ### Vis for jet map ###
    arr_0, arr_1 = np.meshgrid(p_pline.arr_x_cb, p_pline.arr_y_cb)
    rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
    height, width = np.shape(rdr_cube_bev)
    figsize = (1*magnifying, height/width*magnifying) if height>=width else (width/height*magnifying, 1*magnifying)
    plt.figure(figsize=figsize)
    plt.pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig('./resources/imgs/img_cube_bev.png', pad_inches=0, dpi=300) #, plt.show()
    img_jet = cv2.imread('./resources/imgs/img_cube_bev.png')
    img_jet = np.flip(img_jet, axis=0)
    img_jet_vis = cv2.resize(img_jet, (num_x*magnifying, num_y*magnifying))
    ### Vis for jet map ###

    ### Vis for sliced  ###
    if (len(bboxes) == 0) or (bboxes is None):
        pass
        # cv2.imshow('jet map (cv2)', rdr_cube_bev_vis)
        # cv2.imshow('jet map (plt)', img_jet_vis)
    else:
        img_jet_slice = cv2.resize(img_jet, (num_x, num_y)).copy()
        for idx_bbox, bbox in enumerate(bboxes):
            _, _, [x, y, z, theta, xl, yl, zl], _ = bbox
            obj3d = Object3D(x, y, z, xl, yl, zl, theta)
            pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
            list_pts_indices = []
            for pt in pts:
                idx_x = np.argmin(np.abs(p_pline.arr_x_cb-pt[0]))
                idx_y = np.argmin(np.abs(p_pline.arr_y_cb-pt[1]))
                list_pts_indices.append([idx_x, idx_y])
            ### Slicing ###
            arr_indices = np.array(list_pts_indices)
            # print(arr_indices)
            x_min, y_min = np.min(arr_indices, axis=0)
            x_max, y_max = np.max(arr_indices, axis=0)
            list_pts_convex = [(x_min,y_min), (x_min,y_max), (x_max,y_max), (x_max,y_min), (x_min,y_min)]
            cv2.imshow(f'sliced (bev) {idx_bbox}', cv2.resize(img_jet_slice[y_min:y_max,x_min:x_max,:], (0,0), fx=magnifying, fy=magnifying))
            for idx_line in range(4):
                img_jet_slice = cv2.line(img_jet_slice, list_pts_convex[idx_line], list_pts_convex[idx_line+1], (0,0,255), 1)
            cv2.putText(img_jet_slice, f'{idx_bbox}', (x_max,y_min), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))

            sliced_cube = rdr_cube[:,y_min:y_max,x_min:x_max]
            
            ### z for vis ###
            z_min, z_max = 75-20, 75+20
            ### z for vis ###

            ### zx (side view) ###
            arr_0, arr_1 = np.meshgrid(p_pline.arr_x_cb[x_min:x_max], p_pline.arr_z_cb[z_min:z_max])
            sliced_cube_zx = np.mean(sliced_cube, axis=1)
            plt.figure()
            plt.pcolormesh(arr_0, arr_1, sliced_cube_zx[z_min:z_max,:], cmap='jet')
            plt.title(f'object {idx_bbox} zx shape (side view)')
            plt.colorbar()
            ### zy (front view) ###
            arr_0, arr_1 = np.meshgrid(p_pline.arr_y_cb[y_min:y_max], p_pline.arr_z_cb[z_min:z_max])
            sliced_cube_zy = np.mean(sliced_cube, axis=2)
            plt.figure()
            plt.pcolormesh(arr_0, arr_1, sliced_cube_zy[z_min:z_max,:], cmap='jet')
            plt.title(f'object {idx_bbox} zy shape (front view)')
            plt.colorbar()
            # print(sliced_cube.shape)
            ### Slicing ###
    ### Vis for sliced  ###

    if idx_custom_slice is not None:
        if (len(bboxes) == 0) or (bboxes is None):
            img_jet_slice = cv2.resize(img_jet, (num_x, num_y)).copy()
        x_min, x_max, y_min, y_max = idx_custom_slice
        list_pts_convex = [(x_min,y_min), (x_min,y_max), (x_max,y_max), (x_max,y_min), (x_min,y_min)]
        cv2.imshow(f'sliced (bev) custom', cv2.resize(img_jet_slice[y_min:y_max,x_min:x_max,:], (0,0), fx=magnifying, fy=magnifying))
        for idx_line in range(4):
            img_jet_slice = cv2.line(img_jet_slice, list_pts_convex[idx_line], list_pts_convex[idx_line+1], (0,0,255), 1)
        cv2.putText(img_jet_slice, f'custom', (x_max,y_min), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))

        sliced_cube = rdr_cube[:,y_min:y_max,x_min:x_max]
        
        ### z for vis ###
        z_min, z_max = 75-20, 75+20
        ### z for vis ###

        ### zx (side view) ###
        arr_0, arr_1 = np.meshgrid(p_pline.arr_x_cb[x_min:x_max], p_pline.arr_z_cb[z_min:z_max])
        sliced_cube_zx = np.mean(sliced_cube, axis=1)
        plt.figure()
        plt.pcolormesh(arr_0, arr_1, sliced_cube_zx[z_min:z_max,:], cmap='jet')
        plt.title(f'slice custom zx shape (side view)')
        plt.colorbar()
        ### zy (front view) ###
        arr_0, arr_1 = np.meshgrid(p_pline.arr_y_cb[y_min:y_max], p_pline.arr_z_cb[z_min:z_max])
        sliced_cube_zy = np.mean(sliced_cube, axis=2)
        plt.figure()
        plt.pcolormesh(arr_0, arr_1, sliced_cube_zy[z_min:z_max,:], cmap='jet')
        plt.title(f'slice zy shape (front view)')
        plt.colorbar()

    ### Vis for sliced  ###r
    cv2.imshow('jet map with sliced convex', cv2.resize(img_jet_slice, (0,0), fx=magnifying, fy=magnifying))
    cv2.imshow('jet map (cv2)', rdr_cube_bev_vis)
    cv2.imshow('jet map (plt)', img_jet_vis)

    arr_0, arr_1 = np.meshgrid(p_pline.arr_x_cb, p_pline.arr_y_cb)
    plt.figure()
    plt.pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
    # cv2.waitKey(0)
    plt.show()

def func_show_rdr_pc_cube(p_pline, dict_item, bboxes=None, cfar_params = [25, 8, 0.01], axis='x', is_with_lidar=True):
    rdr_cube, _, _ = p_pline.get_cube(dict_item['meta']['path_rdr_cube'], mode=0)
    rdr_pc = get_rdr_pc_from_cube(p_pline, rdr_cube, cfar_params[0], cfar_params[1], cfar_params[2], axis)        
    rdr_pcd = o3d.geometry.PointCloud()
    rdr_pcd.points = o3d.utility.Vector3dVector(rdr_pc)
    rdr_pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(rdr_pc))

    list_vis = [rdr_pcd]

    if is_with_lidar:
        pc_lidar = dict_item['ldr_pc_64']
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
        list_vis.append(pcd)
    
    if not (bboxes is None):
        bboxes_o3d = []
        for obj in bboxes:
            cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
            # try item()
            bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta))

            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_bbox = [p_pline.cfg.CLASS_RGB[cls_name] for _ in range(len(lines))]

        line_sets_bbox = []
        for gt_obj in bboxes_o3d:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors_bbox)
            line_sets_bbox.append(line_set)
        list_vis.extend(line_sets_bbox)

    o3d.visualization.draw_geometries(list_vis)

def func_show_rdr_pc_tesseract(p_pline, dict_item, bboxes=None, cfar_params = [25, 8, 0.01], \
                                roi_x=[0, 100], roi_y=[-50, 50], roi_z=[-10, 10], is_with_lidar=True):
    num_train, num_guard, rate_fa = cfar_params
    pc_radar = get_rdr_pc_from_tesseract(p_pline, dict_item['rdr_tesseract'], num_train, num_guard, rate_fa)
    pc_radar = pc_radar[
        np.where(
            (pc_radar[:, 0] > roi_x[0]) & (pc_radar[:, 0] < roi_x[1]) &
            (pc_radar[:, 1] > roi_y[0]) & (pc_radar[:, 1] < roi_y[1]) &
            (pc_radar[:, 2] > roi_z[0]) & (pc_radar[:, 2] < roi_z[1])
        )
    ]

    num_points, _ = pc_radar.shape
    print(f'number of points = {num_points}')
    
    rdr = get_pc_for_vis(pc_radar, 'black')
    line_sets_bbox = get_bbox_for_vis(bboxes)
    if is_with_lidar:
        pc_lidar = dict_item['ldr_pc_64']
        # pc_lidar = pc_lidar[
        #     np.where(
        #         (pc_lidar[:, 0] > roi_x[0]) & (pc_lidar[:, 0] < roi_x[1]) &
        #         (pc_lidar[:, 1] > roi_y[0]) & (pc_lidar[:, 1] < roi_y[1]) &
        #         (pc_lidar[:, 2] > roi_z[0]) & (pc_lidar[:, 2] < roi_z[1])
        #     )
        # ]
        ldr = get_pc_for_vis(pc_lidar)
        o3d.visualization.draw_geometries([rdr, ldr] + line_sets_bbox)
    else:
        o3d.visualization.draw_geometries([rdr] + line_sets_bbox)

def func_save_undistorted_camera_imgs_w_projected_params(p_ds, dict_args=None, vis=False, save=True):
    if dict_args is None:
        dict_args = dict(
            option = dict(
                undistort = True,
                to_radar_coord = True,
                list_process_cams = ['front0','front1','right0','right1','rear0','rear1','left0','left1'],
                process = dict( # resize -> crop
                    is_process=True,
                    ori_shape=(1280,720), # W, H
                    resize=0.7,
                    crop=(96,170,800,426), # final img shape = (256, 704)
                    flip=False,
                    rotate=0.,
                    is_normalize=True,
                ),
                normalize = dict(
                    mean=[0.303, 0.303, 0.307], # rgb (tools/calc_rgb_distribution)
                    std=[0.113, 0.119, 0.107]
                )
                # normalize = dict(
                #     mean=[0.485, 0.456, 0.406], # from nuscenes
                #     std=[0.229, 0.224, 0.225]
                # )
            ),
            dir_save = dict(
                img_undistorted = '/media/donghee/HDD_0/kradar_imgs/undistorted',
                img_cropped = '/media/donghee/HDD_0/kradar_imgs/cropped',
                # img_processed = '/media/donghee/HDD_0/kradar_imgs/norm_kradar', # norm_nuscenes
                dict_calib = './resources/cam_calib/T_params_seq'
            ),
        )
    
    import pickle
    from easydict import EasyDict
    import torch
    import torchvision
    from tqdm import tqdm
    import copy
    from PIL import Image
    
    dict_args = EasyDict(dict_args)
    dict_option = dict_args.option
    dict_save = dict_args.dir_save

    dict_process = dict_option.process
    is_process = dict_process.is_process
    is_normalize = dict_process.is_normalize

    if is_normalize:
        mean = dict_option.normalize.mean
        std = dict_option.normalize.std
        compose = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    
    to_radar_coord = dict_option.to_radar_coord
    ego_coord_sensor = 'radar' if to_radar_coord else 'lidar'

    prev_seq = [-1 for _ in range(len(dict_option.list_process_cams))]

    dict_calib_cam = dict()
    for idx_seq in range(58):
        seq = f'{idx_seq+1}'
        dict_calib_cam[seq] = dict()

        for key_cam in dict_option.list_process_cams:
            dict_calib_cam[seq][key_cam] = dict()

    for idx_seq in range(58):
        seq = f'{idx_seq+1}'
        
        os.makedirs(osp.join(dict_save.img_undistorted, seq), exist_ok=True)
        for key_cam in dict_option.list_process_cams:
            os.makedirs(osp.join(dict_save.img_undistorted, seq, key_cam), exist_ok=True)

        os.makedirs(osp.join(dict_save.img_cropped, seq), exist_ok=True)
        for key_cam in dict_option.list_process_cams:
            os.makedirs(osp.join(dict_save.img_cropped, seq, key_cam), exist_ok=True)

        # os.makedirs(osp.join(dict_save.img_processed, seq), exist_ok=True)
        # for key_cam in dict_option.list_process_cams:
        #     os.makedirs(osp.join(dict_save.img_processed, seq, key_cam), exist_ok=True)

        # os.makedirs(osp.join(dict_save.dict_calib, seq), exist_ok=True)
    
    if is_process:
        resize = dict_process.resize
        crop = dict_process.crop
        flip = dict_process.flip
        rotate = dict_process.rotate

        rotation = torch.eye(2)
        translation = torch.zeros(2)
        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
        transform = torch.eye(4)
        transform[:2, :2] = rotation
        transform[:2, 3] = translation
        transform = transform.numpy()

        W, H = dict_process.ori_shape
        resize_dims = (int(W * resize), int(H * resize))
    
    for idx_item in tqdm(range(len(p_ds.list_dict_item))):
        dict_item = p_ds.__getitem__(idx_item)
        dict_meta = dict_item['meta']
        
        seq = dict_meta['seq']
        dict_idx = dict_meta['idx']
        dict_path = dict_meta['path']

        calib_l2r = dict_meta['calib']

        for idx_cam, key_cam in enumerate(dict_option.list_process_cams):
            img_temp = dict_item[key_cam]

            ### Calibration ###
            # get calibration for new sequence
            if prev_seq[idx_cam] != int(seq):
                print(f'* new sequence: {seq} / cams: {key_cam}')
                # Should deepcopy variables of self (prevent recursive calculation of pointer)
                img_size_ret, intrinsics_ret, distortion_ret, T_ldr2cam_ret = p_ds.dict_cam_calib[seq][key_cam]

                img_size = copy.deepcopy(img_size_ret)
                intrinsics = copy.deepcopy(intrinsics_ret)
                distortion = copy.deepcopy(distortion_ret)
                T_ldr2cam = copy.deepcopy(T_ldr2cam_ret)
                
                if dict_option.undistort:
                    ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
                
                    for j in range(3):
                        for i in range(3):
                            intrinsics[j,i] = ncm[j, i]
                    
                    map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1)
                    img_temp = cv2.remap(img_temp, map_x, map_y, cv2.INTER_LINEAR)

                T_cam2pix = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0)
                T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0)  # 4x4

                if to_radar_coord:
                    dx, dy, dz = calib_l2r
                    T_ldr2rdr = np.eye(4)
                    T_ldr2rdr[0,3] = dx
                    T_ldr2rdr[1,3] = dy
                    T_ldr2rdr[2,3] = dz

                    T_rdr2ldr = np.linalg.inv(T_ldr2rdr)
                    # print(T_rdr2ldr)

                    T_sen2cam = T_ldr2cam@T_rdr2ldr
                    # print(T_sen2cam)
                else:
                    T_sen2cam = T_ldr2cam

                T_cam2sen = np.linalg.inv(T_sen2cam)                    # 4x4
                T_sen2pix = T_cam2pix@T_sen2cam                         # 4x4

                if is_process:
                    dict_calib_cam[seq][key_cam]['img_aug_matrix'] = transform
                #     print(dict_calib_cam[seq][key_cam]['img_aug_matrix'])

                # print(T_sen2pix)
                # print(T_cam2sen)
                # print(T_sen2cam)
                # print(T_sen2cam@T_cam2sen)

                dict_calib_cam[seq][key_cam]['camera_intrinsics'] = T_cam2pix
                dict_calib_cam[seq][key_cam][ego_coord_sensor+'2image'] = T_sen2pix
                dict_calib_cam[seq][key_cam][ego_coord_sensor+'2camera'] = T_sen2cam
                dict_calib_cam[seq][key_cam]['camera2'+ego_coord_sensor] = T_cam2sen

                ### Save dict_calib_cam ###
                if save:
                    with open(osp.join(dict_save.dict_calib, seq), 'wb') as pickle_file:
                        pickle.dump(dict_calib_cam[seq], pickle_file)
                ### Save dict_calib_cam ###

                prev_seq[idx_cam] = int(seq)

            ### Ori img -> processed img ###
            img_undistorted = Image.fromarray(np.flip(img_temp, axis=2)) # bgr to rgb
            # print(resize_dims)
            img_process = img_undistorted.resize(resize_dims)
            img_process = img_process.crop(crop)
            # print(img_process.size)
            # plt.imshow(img_process)
            # plt.show()

            img_cropped = img_process.copy()
            img_process = compose(img_process)

            # img_process = np.array(img_process).transpose(1,2,0)
            # plt.imshow(img_process)
            # plt.show()
            ### Ori img -> processed img ###
            
            if save:
                # print(dict_idx.keys())
                cam_idx = dict_idx['camf'] # lrr are synchronized to camf
                img_undistorted.save(osp.join(dict_save.img_undistorted, seq, key_cam, f'{cam_idx}.png'))
                img_cropped.save(osp.join(dict_save.img_cropped, seq, key_cam, f'{cam_idx}.png'))
                # np.save(osp.join(dict_save.img_processed, seq, key_cam, f'{cam_idx}.npy'), np.array(img_process))

            if vis:
                if to_radar_coord:
                    ldr_points = p_ds.get_ldr64(dict_item)['ldr64']
                else:
                    ldr_points = p_ds.get_ldr64_from_path(dict_path['ldr64'], is_calib=False) # wo_calib
                
                ### Ori ###
                pc_ldr = (np.insert(ldr_points[:,:3], 3, [1], axis=1)).T
                pc_cam = dict_calib_cam[seq][key_cam][ego_coord_sensor+'2image']@pc_ldr
                pc_cam[:2,:] /= pc_cam[2,:]
                
                plt.figure(figsize=(20,15),dpi=96,tight_layout=True)
                img_w,img_h = img_undistorted.size
                plt.axis([0,img_w,img_h,0])
                plt.imshow(img_undistorted)
                pc_cam = (pc_cam.T)[:,:3]
                pc_cam = pc_cam[np.where(
                    (pc_cam[:,0]>=0) & (pc_cam[:,0]<img_w) &
                    (pc_cam[:,1]>=0) & (pc_cam[:,1]<img_h) &
                    (pc_cam[:,2]>3))]
                
                plt.scatter(pc_cam[:,0],pc_cam[:,1],c=1/pc_cam[:,2],cmap='rainbow_r',alpha=0.5,s=10.0)
                plt.xticks([])
                plt.yticks([])
                ### Ori ###

                ### Resize & Cropped ###
                if is_process:
                    pc_ldr = (np.insert(ldr_points[:,:3], 3, [1], axis=1)).T
                    pc_cam = dict_calib_cam[seq][key_cam][ego_coord_sensor+'2image']@pc_ldr
                    pc_cam[:2,:] /= pc_cam[2,:]
                    pc_cam = dict_calib_cam[seq][key_cam]['img_aug_matrix']@pc_cam
                    
                    plt.figure(figsize=(20,13),dpi=96,tight_layout=True)
                    img_w,img_h = img_cropped.size
                    plt.axis([0,img_w,img_h,0])
                    plt.imshow(img_cropped)
                    pc_cam = (pc_cam.T)[:,:3]
                    pc_cam = pc_cam[np.where(
                        (pc_cam[:,0]>=0) & (pc_cam[:,0]<img_w) &
                        (pc_cam[:,1]>=0) & (pc_cam[:,1]<img_h) &
                        (pc_cam[:,2]>3))]
                    
                    plt.scatter(pc_cam[:,0],pc_cam[:,1],c=1/pc_cam[:,2],cmap='rainbow_r',alpha=0.5,s=10.0)
                    plt.xticks([])
                    plt.yticks([])

                plt.show()
                plt.close()
                ### Ori ###

        # free memory (Killed error, checked with htop)
        for k in dict_item.keys():
            if k != 'meta':
                dict_item[k] = None

def func_get_distribution_of_label(p_ds, consider_avail=True):
    from tqdm import tqdm

    dict_label = p_ds.label.copy()
    dict_label.pop('calib')
    dict_label.pop('onlyR')
    dict_label.pop('Label')
    dict_label.pop('consider_cls')
    dict_label.pop('consider_roi')
    dict_label.pop('remove_0_obj')
    
    dict_for_dist = dict()
    dict_for_value = dict()
    dict_for_min_xyz = dict()
    dict_for_max_xyz = dict()
    for obj_name in dict_label.keys():
        dict_for_dist[obj_name] = 0
        dict_for_value[obj_name] = [0., 0., 0.]
        dict_for_min_xyz[obj_name] = [10000., 10000., 10000.]
        dict_for_max_xyz[obj_name] = [-10000., -10000., -10000.]
    
    if consider_avail:
        dict_avail = dict()
        list_avails = ['R', 'L', 'L1']
        for avail in list_avails:
            dict_temp = dict()
            for obj_name in dict_label.keys():
                dict_temp[obj_name] = 0
            dict_avail[avail] = dict_temp

    for dict_item in tqdm(p_ds.list_dict_item):
        dict_item = p_ds.get_label(dict_item)
        for obj in dict_item['meta']['label']:
            cls_name, (x, y, z, th, l, w, h), trk, avail = obj
            dict_for_dist[cls_name] += 1
            dict_for_value[cls_name][0] += l
            dict_for_value[cls_name][1] += w
            dict_for_value[cls_name][2] += h
            dict_for_min_xyz[cls_name][0] = min(dict_for_min_xyz[cls_name][0], x)
            dict_for_max_xyz[cls_name][0] = max(dict_for_max_xyz[cls_name][0], x)
            dict_for_min_xyz[cls_name][1] = min(dict_for_min_xyz[cls_name][1], y)
            dict_for_max_xyz[cls_name][1] = max(dict_for_max_xyz[cls_name][1], y)
            dict_for_min_xyz[cls_name][2] = min(dict_for_min_xyz[cls_name][2], z)
            dict_for_max_xyz[cls_name][2] = max(dict_for_max_xyz[cls_name][2], z)

            # if x<0:
            #     print(x)

            try:
                if consider_avail:
                    dict_avail[avail][cls_name] += 1
            except:
                print(dict_item['meta']['label_v2_1'])

    for obj_name in dict_for_dist.keys():
        n_obj = dict_for_dist[obj_name]
        l, w, h = dict_for_value[obj_name]
        min_x, min_y, min_z = dict_for_min_xyz[obj_name]
        max_x, max_y, max_z = dict_for_max_xyz[obj_name]
        print('* # of ', obj_name, ': ', n_obj)
        divider = np.maximum(n_obj, 1)
        print('* lwh of ', obj_name, ': ', l/divider, ', ', w/divider, ', ', h/divider)
        print('* min xyz of ', obj_name, ': ', min_x, ', ', min_y, ', ', min_z)
        print('* max xyz of ', obj_name, ': ', max_x, ', ', max_y, ', ', max_z)
    
    if consider_avail:
        for avail in list_avails:
            print('-'*30, avail, '-'*30)
            for obj_name in dict_avail[avail].keys():
                print('* # of ', obj_name, ': ', dict_avail[avail][obj_name])
