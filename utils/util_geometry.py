"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.12.29
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: utils for geometry
"""

import numpy as np
import cv2
import open3d as o3d

__all__ = [
    'get_xy_from_ra_color',
    'draw_bbox_in_yx_bgr',
    'get_2d_gaussian_kernel',
    'get_gaussian_confidence_cart',
    'change_arr_cart_to_polar_2d',
    'get_high_resolution_array',
    'get_rdr_pc_from_cube',
    'get_rdr_pc_from_tesseract',
    'get_pc_for_vis',
    'get_bbox_for_vis',
    'Object3D',
    ]

def get_xy_from_ra_color(ra_in, arr_range_in, arr_azimuth_in, roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_in_deg=False):
    '''
    * args:
    *   roi_x = [min_x, bin_x, max_x] [m]
    *   roi_y = [min_y, bin_y, max_y] [m]
    '''
    
    if len(ra_in.shape) == 2:
        num_range, num_azimuth = ra_in.shape
    elif len(ra_in.shape) == 3:
        num_range, num_azimuth, _ = ra_in.shape

    assert (num_range == len(arr_range_in) and num_azimuth == len(arr_azimuth_in))
    
    ra = ra_in.copy()
    arr_range = arr_range_in.copy()
    arr_azimuth = arr_azimuth_in.copy()

    if is_in_deg:
        arr_azimuth = arr_azimuth*np.pi/180.
    
    min_x, bin_x, max_x = roi_x
    min_y, bin_y, max_y = roi_y

    # with +1
    arr_x = np.linspace(min_x, max_x, int((max_x-min_x)/bin_x)+1)
    arr_y = np.linspace(min_y, max_y, int((max_y-min_y)/bin_y)+1)
    # print(arr_x), print(arr_y)

    max_r = np.max(arr_range)
    min_r = np.min(arr_range)

    max_azi = np.max(arr_azimuth)
    min_azi = np.min(arr_azimuth)

    num_y = len(arr_y)
    num_x = len(arr_x)
    
    arr_yx = np.zeros((num_y, num_x, 3), dtype=np.uint8)

    # print(f'arr_yx shape: {arr_yx.shape}')

    # Inverse warping
    for idx_y, y in enumerate(arr_y):
        for idx_x, x in enumerate(arr_x):
            # bilinear interpolation

            # print(f'y, x = {y}, {x}')
            r = np.sqrt(x**2 + y**2)
            azi = np.arctan2(-y,x) # for real physical azimuth
            # azi = np.arctan2(y,x)
            # print(f'r, azi = {r}, {azi}')
            
            if (r < min_r) or (r > max_r) or (azi < min_azi) or (azi > max_azi):
                continue
            
            try:
                idx_r_0, idx_r_1 = find_nearest_two(r, arr_range)
                idx_a_0, idx_a_1 = find_nearest_two(azi, arr_azimuth)
            except:
                continue

            if (idx_r_0 == -1) or (idx_r_1 == -1) or (idx_a_0 == -1) or (idx_a_1 == -1):
                continue
            
            ra_00 = ra[idx_r_0,idx_a_0,:]
            ra_01 = ra[idx_r_0,idx_a_1,:]
            ra_10 = ra[idx_r_1,idx_a_0,:]
            ra_11 = ra[idx_r_1,idx_a_1,:]

            val = (ra_00*(arr_range[idx_r_1]-r)*(arr_azimuth[idx_a_1]-azi)\
                    +ra_01*(arr_range[idx_r_1]-r)*(azi-arr_azimuth[idx_a_0])\
                    +ra_10*(r-arr_range[idx_r_0])*(arr_azimuth[idx_a_1]-azi)\
                    +ra_11*(r-arr_range[idx_r_0])*(azi-arr_azimuth[idx_a_0]))\
                    /((arr_range[idx_r_1]-arr_range[idx_r_0])*(arr_azimuth[idx_a_1]-arr_azimuth[idx_a_0]))

            arr_yx[idx_y, idx_x] = val

    return arr_yx, arr_y, arr_x

def find_nearest_two(value, arr):
    '''
    * args
    *   value: float, value in arr
    *   arr: np.array
    * return
    *   idx0, idx1 if is_in_arr else -1
    '''
    arr_temp = arr - value
    arr_idx = np.argmin(np.abs(arr_temp))
    
    try:
        if arr_temp[arr_idx] < 0: # min is left
            if arr_temp[arr_idx+1] < 0:
                return -1, -1
            return arr_idx, arr_idx+1
        elif arr_temp[arr_idx] >= 0:
            if arr_temp[arr_idx-1] >= 0:
                return -1, -1
            return arr_idx-1, arr_idx
    except:
        return -1, -1

def draw_bbox_in_yx_bgr(arr_yx_in, arr_y_in, arr_x_in, label_in, is_with_bbox_mask=True, lthick=1):
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

    dic_cls_bgr = {
        'Sedan': [23,208,253],
        'Bus or Truck': [0,50,255],
        'Motorcycle': [0,0,255],
        'Bicycle': [0,255,255],
        'Pedestrian': [255,0,0],
        'Pedestrian Group': [255,0,100],
        'Label': [128,128,128],
        'Infer': [0,50,250],
        'Gt Sedan': [23,208,253]
    }

    for obj in label:
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj

        # calib is already reflected in dataset
        color = dic_cls_bgr[cls_name]
        
        x_pix = (x-x_min)/x_m_per_pix
        y_pix = (y-y_min)/y_m_per_pix
        # y_pix = (-y-y_min)/y_m_per_pix
        
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

        arr_yx = cv2.line(arr_yx, pts[0], pts[1], color, lthick)
        arr_yx = cv2.line(arr_yx, pts[1], pts[2], color, lthick)
        arr_yx = cv2.line(arr_yx, pts[2], pts[3], color, lthick)
        arr_yx = cv2.line(arr_yx, pts[3], pts[0], color, lthick)

        # front and center
        pt_cen = (int(np.around(x_pix)), int(np.around(y_pix)))
        # arr_yx = cv2.line(arr_yx, pt_cen, pt_front, color, 1)

        # make center black
        arr_yx = cv2.circle(arr_yx, pt_cen, 1, (0,0,0), thickness=-1)

    return arr_yx

class Object3D():
    def __init__(self, xc, yc, zc, xl, yl, zl, rot_rad):
        self.xc, self.yc, self.zc, self.xl, self.yl, self.zl, self.rot_rad = xc, yc, zc, xl, yl, zl, rot_rad
        
        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 

        self.corners = np.row_stack((corners_x, corners_y, corners_z))
    
        rotation_matrix = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0.0],
            [np.sin(rot_rad), np.cos(rot_rad), 0.0],
            [0.0, 0.0, 1.0]])
        
        self.corners = rotation_matrix.dot(self.corners).T + np.array([[self.xc, self.yc, self.zc]])


def get_2d_gaussian_kernel(pix_h, pix_w, pr_sigma=0.15, is_normalize_to_1=True):
    sigma_h = pix_h*pr_sigma
    sigma_w = pix_w*pr_sigma

    kernel1d_h = cv2.getGaussianKernel(int(np.around(pix_h)), sigma_h)
    kernel1d_w = cv2.getGaussianKernel(int(np.around(pix_w)), sigma_w)
    kernel2d = np.outer(kernel1d_h, kernel1d_w.transpose())

    if is_normalize_to_1:
        kernel2d = kernel2d/np.max(kernel2d)

    return kernel2d
    
def get_gaussian_confidence_cart(roi_x, roi_y, bboxes=None, \
                                    is_vis=False, is_for_bbox_vis=False):
    min_x, bin_x, max_x = roi_x
    min_y, bin_y, max_y = roi_y

    # with +1
    arr_x = np.linspace(min_x, max_x, int((max_x-min_x)/bin_x)+1)
    arr_y = np.linspace(min_y, max_y, int((max_y-min_y)/bin_y)+1)

    y_min = np.min(arr_y)
    x_min = np.min(arr_x)

    num_y = len(arr_y)
    num_x = len(arr_x)
    
    if is_vis:
        arr_yx_vis = np.zeros((num_y, num_x, 3), dtype=np.uint8)

    arr_yx_conf = np.zeros((num_y, num_x), dtype=float)

    y_m_per_pix = np.mean(arr_y[1:] - arr_y[:-1])
    x_m_per_pix = np.mean(arr_x[1:] - arr_x[:-1])

    for idx_iter, obj in enumerate(bboxes):
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
        
        x_pix = (x-x_min)/x_m_per_pix
        y_pix = (y-y_min)/y_m_per_pix
        
        l_pix = l/x_m_per_pix
        w_pix = w/y_m_per_pix
        
        ### BBox in Cartesian ###
        # pts = [ [l_pix/2, w_pix/2],
        #         [l_pix/2, -w_pix/2],
        #         [-l_pix/2, -w_pix/2],
        #         [-l_pix/2, w_pix/2] ]
        # # front left, front right, back right, back left

        # cos_th = np.cos(theta)
        # sin_th = np.sin(theta)
        # # rotation & translation
        # pts = list(map(lambda pt: [ x_pix+cos_th*pt[0]-sin_th*pt[1], \
        #                             y_pix+sin_th*pt[0]+cos_th*pt[1] ], pts))
        # draw_bbox_2d(arr_yx, pts, [x_pix, y_pix])
        ### BBox in Cartesian ###

        ### BBox (Rotated) ###
        # pts = [ [l_pix/2, w_pix/2],
        #         [l_pix/2, -w_pix/2],
        #         [-l_pix/2, -w_pix/2],
        #         [-l_pix/2, w_pix/2] ]
        # cos_th = np.cos(theta)
        # sin_th = np.sin(theta)
        # pts_only_rot = list(map(lambda pt: [ cos_th*pt[0]-sin_th*pt[1], \
        #                                      sin_th*pt[0]+cos_th*pt[1] ], pts))
        # x_tr, y_tr = np.min(np.array(pts_only_rot), axis=0)
        # x_cen = int(np.around(-x_tr))
        # y_cen = int(np.around(-y_tr))

        # pts_only_rot = list(map(lambda pt: [pt[0]-x_tr, pt[1]-y_tr], pts_only_rot))
        # x_total, y_total = np.max(np.array(pts_only_rot), axis=0)
        # bbox_total = np.zeros((int(np.ceil(y_total)), int(np.ceil(x_total)), 3), dtype=np.uint8)
        
        # draw_bbox_2d(bbox_total, pts_only_rot, [x_cen, y_cen], line_color=[255,255,255], cts_color=[0,0,0])
        # cv2.imshow('bbox_total', bbox_total)
        ### BBox (Rotated) ###

        kernel_2d = get_2d_gaussian_kernel(w_pix, l_pix)

        ### quater ###
        # Bounding box length translation first
        # M = np.float32([[1.,0.,-l_pix/2.],[0.,1.,-w_pix/2.]])
        # kernel_2d_affine = cv2.warpAffine(kernel_2d, M, (num_x, num_y))
        
        # cos_th = np.cos(theta)
        # sin_th = np.sin(theta)
        
        # M = np.float32([[cos_th,-sin_th,x_pix],[sin_th,cos_th,y_pix]])
        # kernel_2d_affine = cv2.warpAffine(kernel_2d_affine, M, (num_x, num_y))
        ### quater ###

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        M = np.float32([[cos_th,-sin_th,x_pix-l_pix/2.*cos_th+w_pix/2.*sin_th],\
                        [sin_th,cos_th,y_pix-l_pix/2.*sin_th-w_pix/2.*cos_th]])
        kernel_2d_affine = cv2.warpAffine(kernel_2d, M, (num_x, num_y))

        arr_yx_conf += kernel_2d_affine

        if is_vis:
            cv2.imshow(f'kernel_2d_{idx_iter}', kernel_2d)

            pts = [ [l_pix/2, w_pix/2],
                    [l_pix/2, -w_pix/2],
                    [-l_pix/2, -w_pix/2],
                    [-l_pix/2, w_pix/2] ]
            # front left, front right, back right, back left

            # rotation & translation
            pts = list(map(lambda pt: [ x_pix+cos_th*pt[0]-sin_th*pt[1], \
                                        y_pix+sin_th*pt[0]+cos_th*pt[1] ], pts))
            
            kernel_2d_affine_vis = cv2.cvtColor((kernel_2d_affine*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            draw_bbox_2d(kernel_2d_affine_vis, pts, [x_pix, y_pix])
            
            # Flip b4 show (Cartesian)
            kernel_2d_affine_vis = kernel_2d_affine_vis.transpose((1,0,2))
            kernel_2d_affine_vis = np.flip(kernel_2d_affine_vis, axis=(0,1))

            cv2.imshow(f'kernel_2d_affine_{idx_iter}', kernel_2d_affine_vis)

    arr_yx_conf = np.clip(arr_yx_conf, 0., 1.)

    if is_for_bbox_vis:
        return arr_yx_conf, arr_y, arr_x
    else:
        return arr_yx_conf

def draw_bbox_2d(arr_yx, pts, cts, line_color=[0,0,255], cts_color=[0,255,0]):
    x_pix, y_pix = cts
    pt_front = (int(np.around((pts[0][0]+pts[1][0])/2)), int(np.around((pts[0][1]+pts[1][1])/2)))
    
    # to integer and tuple
    pts = list(map(lambda pt: (int(np.around(pt[0])), int(np.around(pt[1]))), pts))

    arr_yx = cv2.line(arr_yx, pts[0], pts[1], line_color, 1)
    arr_yx = cv2.line(arr_yx, pts[1], pts[2], line_color, 1)
    arr_yx = cv2.line(arr_yx, pts[2], pts[3], line_color, 1)
    arr_yx = cv2.line(arr_yx, pts[3], pts[0], line_color, 1)

    # front and center
    pt_cen = (int(np.around(x_pix)), int(np.around(y_pix)))
    arr_yx = cv2.line(arr_yx, pt_cen, pt_front, line_color, 1)

    # make center black
    arr_yx = cv2.circle(arr_yx, pt_cen, 1, cts_color, thickness=-1)

def change_arr_cart_to_polar_2d(arr_yx, roi_x, roi_y, arr_range, arr_azimuth, dtype='float'):
    if dtype == 'float':
        return change_arr_cart_to_polar_2d_float(arr_yx, roi_x, roi_y, arr_range, arr_azimuth)
    elif dtype == 'color':
        return change_arr_cart_to_polar_2d_color(arr_yx, roi_x, roi_y, arr_range, arr_azimuth)

def change_arr_cart_to_polar_2d_float(arr_yx, roi_x, roi_y, arr_range, arr_azimuth):
    min_x, bin_x, max_x = roi_x
    min_y, bin_y, max_y = roi_y

    # with +1
    arr_x = np.linspace(min_x, max_x, int((max_x-min_x)/bin_x)+1)
    arr_y = np.linspace(min_y, max_y, int((max_y-min_y)/bin_y)+1)

    len_range = len(arr_range)
    len_azimuth = len(arr_azimuth)

    arr_ra = np.zeros((len_range, len_azimuth), dtype=float)

    for idx_r, r in enumerate(arr_range):
        for idx_a, a in enumerate(arr_azimuth):
            x = r*np.cos(-a)
            y = r*np.sin(-a)

            if (x < min_x) or (x > max_x) or (y < min_y) or (y > max_y):
                continue

            try:
                idx_x_0, idx_x_1 = find_nearest_two(x, arr_x)
                idx_y_0, idx_y_1 = find_nearest_two(y, arr_y)
            except:
                continue

            yx_00 = arr_yx[idx_y_0,idx_x_0]
            yx_01 = arr_yx[idx_y_0,idx_x_1]
            yx_10 = arr_yx[idx_y_1,idx_x_0]
            yx_11 = arr_yx[idx_y_1,idx_x_1]
            
            val = (yx_00*(arr_y[idx_y_1]-y)*(arr_x[idx_x_1]-x)\
                    +yx_01*(arr_y[idx_y_1]-y)*(x-arr_x[idx_x_0])\
                    +yx_10*(y-arr_y[idx_y_0])*(arr_x[idx_x_1]-x)\
                    +yx_11*(y-arr_y[idx_y_0])*(x-arr_x[idx_x_0]))\
                    /((arr_y[idx_y_1]-arr_y[idx_y_0])*(arr_x[idx_x_1]-arr_x[idx_x_0]))

            arr_ra[idx_r, idx_a] = val
    
    return arr_ra

def change_arr_cart_to_polar_2d_color(arr_yx, roi_x, roi_y, arr_range, arr_azimuth):
    min_x, bin_x, max_x = roi_x
    min_y, bin_y, max_y = roi_y

    # with +1
    arr_x = np.linspace(min_x, max_x, int((max_x-min_x)/bin_x)+1)
    arr_y = np.linspace(min_y, max_y, int((max_y-min_y)/bin_y)+1)

    len_range = len(arr_range)
    len_azimuth = len(arr_azimuth)

    arr_ra = np.zeros((len_range, len_azimuth, 3), dtype=np.uint8)

    for idx_r, r in enumerate(arr_range):
        for idx_a, a in enumerate(arr_azimuth):
            x = r*np.cos(-a)
            y = r*np.sin(-a)

            if (x < min_x) or (x > max_x) or (y < min_y) or (y > max_y):
                continue

            try:
                idx_x_0, idx_x_1 = find_nearest_two(x, arr_x)
                idx_y_0, idx_y_1 = find_nearest_two(y, arr_y)
            except:
                continue

            yx_00 = arr_yx[idx_y_0,idx_x_0,:]
            yx_01 = arr_yx[idx_y_0,idx_x_1,:]
            yx_10 = arr_yx[idx_y_1,idx_x_0,:]
            yx_11 = arr_yx[idx_y_1,idx_x_1,:]
            
            val = (yx_00*(arr_y[idx_y_1]-y)*(arr_x[idx_x_1]-x)\
                    +yx_01*(arr_y[idx_y_1]-y)*(x-arr_x[idx_x_0])\
                    +yx_10*(y-arr_y[idx_y_0])*(arr_x[idx_x_1]-x)\
                    +yx_11*(y-arr_y[idx_y_0])*(x-arr_x[idx_x_0]))\
                    /((arr_y[idx_y_1]-arr_y[idx_y_0])*(arr_x[idx_x_1]-arr_x[idx_x_0]))

            arr_ra[idx_r, idx_a] = val
    
    return arr_ra

def get_high_resolution_array(arr, scale):
    if not (type(scale) is int):
        assert True, print('scale should be int')

    len_a = len(arr)
    bin_a = np.mean(arr[1:]-arr[:-1])

    a_new = np.zeros((len_a*scale,), dtype=type(arr[0]))
    
    b_new = bin_a/float(scale)
    for idx, val in enumerate(arr):
        for idx_s in range(scale):
            a_new[scale*idx+idx_s] = val + idx_s*b_new

    return a_new

def cell_avg_cfar(x, num_train, num_guard, rate_fa):
    '''
    * referred to https://github.com/marcelsheeny/radiate_sdk
    '''
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half
    alpha = num_train * (rate_fa**(-1 / num_train) - 1)
    mask = np.ones(num_side * 2)
    mask[num_train_half:num_guard] = 0
    mask /= num_train
    noise = np.convolve(x, mask, 'same')
    threshold = alpha * noise
    thr_idx = np.where(x > threshold)

    return thr_idx

def get_rdr_pc_from_tesseract(p_pline, x, num_train, num_guard, rate_fa, is_cart=True, \
                    is_z_reverse = True, is_with_doppler_value=False, is_with_power_value=False):
    x_3d = np.mean(x, axis=0)
    _, n_a, n_e = x_3d.shape
    list_points = []
    for idx_a in range(n_a):
        for idx_e in range(n_e):
            thr_vec = cell_avg_cfar(x_3d[:,idx_a,idx_e], num_train, num_guard, rate_fa)[0]
            val_a = p_pline.arr_azimuth[idx_a]
            val_e = p_pline.arr_elevation[idx_e]
            for idx_r in thr_vec:
                val_r = p_pline.arr_range[idx_r]
                vec_values = []
                if is_cart:
                    if is_z_reverse:
                        val_z = -val_r*np.sin(val_e)
                    else:
                        val_z = val_r*np.sin(val_e)
                    val_y = val_r*np.cos(val_e)*np.sin(-val_a)
                    val_x = val_r*np.cos(val_e)*np.cos(-val_a)
                    vec_values.extend([val_x, val_y, val_z])
                else:
                    vec_values.extend([val_r, val_a, val_e])
                if is_with_doppler_value:
                    vec_values.extend([p_pline.arr_doppler[np.argmax(x[:,idx_r,idx_a,idx_e])]])
                if is_with_power_value:
                    vec_values.extend([x_3d[idx_r,idx_a,idx_e]])
                list_points.append(vec_values)
    
    return np.array(list_points)

def get_rdr_pc_from_cube_axis_x(p_pline, cube_in, num_train, num_guard, rate_fa):
    '''
    * criterion axis is the axis for cfar, due to 1d cfar
    '''
    n_z, n_y, _ = cube_in.shape
    cube = cube_in.copy()
    list_points = []

    for idx_z in range(n_z):
        for idx_y in range(n_y):
            thr_vec = cell_avg_cfar(cube[idx_z,idx_y,:], num_train, num_guard, rate_fa)[0]
            val_z = p_pline.arr_z_cb[idx_z]
            val_y = p_pline.arr_y_cb[idx_y]
            for idx_x in thr_vec:
                val_x = p_pline.arr_x_cb[idx_x]
                list_points.append([val_x, val_y, val_z])
    
    return np.array(list_points)

def get_rdr_pc_from_cube_axis_y(p_pline, cube_in, num_train, num_guard, rate_fa):
    '''
    * criterion axis is the axis for cfar, due to 1d cfar
    '''
    n_z, _, n_x = cube_in.shape
    cube = cube_in.copy()
    list_points = []

    for idx_z in range(n_z):
        for idx_x in range(n_x):
            thr_vec = cell_avg_cfar(cube[idx_z,:,idx_x], num_train, num_guard, rate_fa)[0]
            val_z = p_pline.arr_z_cb[idx_z]
            val_x = p_pline.arr_x_cb[idx_x]
            for idx_y in thr_vec:
                val_y = p_pline.arr_y_cb[idx_y]
                list_points.append([val_x, val_y, val_z])
    
    return np.array(list_points)

def get_rdr_pc_from_cube_axis_z(p_pline, cube_in, num_train, num_guard, rate_fa):
    '''
    * criterion axis is the axis for cfar, due to 1d cfar
    '''
    _, n_y, n_x = cube_in.shape
    cube = cube_in.copy()
    list_points = []

    for idx_y in range(n_y):
        for idx_x in range(n_x):
            thr_vec = cell_avg_cfar(cube[:,idx_y,idx_x], num_train, num_guard, rate_fa)[0]
            val_y = p_pline.arr_y_cb[idx_y]
            val_x = p_pline.arr_x_cb[idx_x]
            for idx_z in thr_vec:
                val_z = p_pline.arr_x_cb[idx_z]
                list_points.append([val_x, val_y, val_z])
    
    return np.array(list_points)

def get_rdr_pc_from_cube(p_pline, cube_in, num_train, num_guard, rate_fa, axis='x'):
    '''
    * criterion axis is the axis for cfar, due to 1d cfar
    '''
    dict_func = {
        'x': get_rdr_pc_from_cube_axis_x,
        'y': get_rdr_pc_from_cube_axis_y,
        'z': get_rdr_pc_from_cube_axis_z
    }

    return dict_func[axis](p_pline, cube_in, num_train, num_guard, rate_fa)

def get_pc_for_vis(pc, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    num_points, _ = pc.shape
    if color=='black':
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pc[:,:3]))
    elif color=='gray':
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.array([0.8, 0.8, 0.8])[np.newaxis,:], num_points, axis=0))
    elif not (color is None):
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.array(color)[np.newaxis,:], num_points, axis=0))
    
    return pcd

def get_bbox_for_vis(bboxes, cfg=None):
    bboxes_o3d = []
    for obj in bboxes:
        _, _, [x,y,z,theta,l,w,h], _ = obj
        bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta))
    lines = [[0, 1], [2, 3], # [0, 3], [1, 2],
                [4, 5], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7],
                [0, 2], [1, 3], [4, 6], [5, 7]]
    if cfg is None:
        colors_bbox = [[1., 0., 0.] for _ in range(len(lines))]
    else:
        pass # TBD with cfg

    line_sets_bbox = []
    for gt_obj in bboxes_o3d:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors_bbox)
        line_sets_bbox.append(line_set)
    
    return line_sets_bbox
