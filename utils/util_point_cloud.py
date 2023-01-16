"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.10.07
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for object detection labeling
"""

# Library
import os
import numpy as np
import cv2
import open3d as o3d

# User Library
import configs.config_general as cnf
import configs.config_ui as cnf_ui

from utils.util_ui_labeling import get_list_dict_by_processing_plain_text

__all__ = [ 'PointCloudOs64',
            'get_pc_os64_with_path',
            'filter_pc_os64_with_roi',
            'append_image_index_to_pc_os64',
            'get_projection_image_from_pointclouds',
            'get_filtered_point_cloud_from_plain_text',
            'get_front_beside_image_from_point_cloud',
            'get_o3d_point_cloud',
            'filter_pc_with_roi_in_xyz',
            'get_o3d_line_set_from_tuple_bbox',
            'get_points_power_from_cube_bev',
            'get_list_bboxes_tuples_from_inference',
            'get_o3d_line_set_from_list_infos',
            'Object3D' ]

class PointCloudOs64:
    def __init__(self, path_pcd):
        f = open(path_pcd, 'r')
        
        ### header
        lines = f.readlines()
        list_fields_name = []
        for i in range(11):
            line = lines[i].split(' ')
            if line[0] == 'FIELDS':
                for j in range(len(line)-1):
                    list_fields_name.append(line[j+1])
                list_fields_name[-1] = list_fields_name[-1].split('\n')[0]
            elif line[0] == 'POINTS':
                self._points_num = int(line[1])

        ### fields
        self._list_fields_name = list_fields_name
        list_idx = np.arange(11, 11+self._points_num)
        self._points = []
        for i in list_idx:
            line = lines[i].split(' ')
            line = list(map(lambda x: float(x), line))
            self._points.append(line) # x, y, z, intensity, reflectivity, ring

        f.close()
    
    def __getitem__(self, idx):
        return np.array(self._points[idx])

    def fields(self):
        return self._list_fields_name

    def points(self):
        return np.array(self._points)
    
    def points_in_list(self):
        return self._points

def get_pc_os64_with_path(path_pcd, len_header=13):
    '''
    *  in: pcd file, e.g., /media/donghee/T5/MMLDD/train/seq_1/pc/pc_001270427447090.pcd
    * out: Pointcloud dictionary
    *       keys: 'path',   'points',   'fields'
    *       type: str,      np.array,   list 
    '''
    f = open(path_pcd, 'r')
    lines = f.readlines()

    header = lines[:len_header]

    for text in header:
        list_text = text.split(' ')
        if list_text[0] == 'FIELDS':
            list_fields = text.split(' ')[1:]
            list_fields[-1] = list_fields[-1][:-1]
    # num_points = int(header[-2].split(' ')[1])
    points_with_fields = lines[len_header:]

    # assert num_points == len(points_with_fields), \
    #     f'The number of points is not {num_points}'
    
    points_with_fields = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), points_with_fields))
    points_with_fields = np.array(points_with_fields)
    f.close()

    pc = dict()
    pc['path'] = path_pcd
    pc['values'] = points_with_fields
    pc['fields'] = list_fields

    return pc

def filter_pc_os64_with_roi(pc_os64, list_roi, filter_mode='xy'):
    '''
    *  in: Pointcloud dictionary
    *       e.g., list roi xy: [x min, x max, y min, y max], meter in LiDAR coords
    * out: Pointcloud dictionary
    '''
    if filter_mode == 'xy':
        return filter_pc_os64_with_roi_in_xy(pc_os64, list_roi)
    elif filter_mode == 'xyz':
        return filter_pc_os64_with_roi_in_xyz(pc_os64, list_roi)

def filter_pc_os64_with_roi_in_xy(pc_os64, list_roi_xy):
    x_min, x_max, y_min, y_max = list_roi_xy
    list_pc_values = pc_os64['values'].tolist()
    pc_os64['values'] = np.array(list(filter(lambda point: \
        (point[0] > x_min) and (point[0] < x_max) and \
        (point[1] > y_min) and (point[1] < y_max), list_pc_values)))
    
    return pc_os64

def filter_pc_os64_with_roi_in_xyz(pc_os64, list_roi_xyz):
    x_min, x_max, y_min, y_max, z_min, z_max = list_roi_xyz
    list_pc_values = pc_os64['values'].tolist()
    pc_os64['values'] = np.array(list(filter(lambda point: \
        (point[0] > x_min) and (point[0] < x_max) and \
        (point[1] > y_min) and (point[1] < y_max) and \
        (point[2] > z_min) and (point[2] < z_max), list_pc_values)))
    
    return pc_os64

def append_image_index_to_pc_os64(pc_os64, list_roi_xy, list_grid_xy):
    '''
    *  in: Pointcloud dictionary
    *       list roi xy: [x min, x max, y min, y max], meter in LiDAR coords
    *       list grid xy: [x grid, y grid], meter in LiDAR coords
    * out: Pointcloud dictionary
    *       keys: 'path',   'points',   'fields',   'img_coords'
    *       type: str,      np.array,   list,       np.array
    '''
    x_min, _, y_min, _ = list_roi_xy
    x_grid, y_grid = list_grid_xy

    list_xy_values = pc_os64['values'][:,:2].tolist()
    list_xy_values = list(map(lambda xy: [int((xy[0]-x_min)/x_grid), \
                                            int((xy[1]-y_min)/y_grid)], list_xy_values))
    
    # np.where convention
    arr_xy_values = np.array(list_xy_values)
    tuple_xy = (arr_xy_values[:,0], arr_xy_values[:,1])

    pc_os64.update({'img_idx': arr_xy_values})
    pc_os64.update({'img_idx_np_where': tuple_xy})

    return pc_os64

def get_projection_image_from_pointclouds(pc_os64, list_img_size_xy=[1152, 1152], list_value_idx = [2, 3, 4], \
                                            list_list_range = [[-2.0,1.5], [0,128], [0,32768]], is_flip=False):
    '''
    *  in: Pointcloud dictionary with 'img_idx'
    * out: Image
            value: 0 ~ 1 normalized by list range
            type: float
    '''
    n_channels = len(list_value_idx)
    temp_img = np.full((list_img_size_xy[0], list_img_size_xy[1], n_channels), 0, dtype=float)

    list_list_values = [] # z, intensity, reflectivity
    for channel_idx, value_idx in enumerate(list_value_idx):
        temp_arr = pc_os64['values'][:,value_idx].copy()

        # Normalize
        v_min, v_max = list_list_range[channel_idx]
        temp_arr[np.where(temp_arr<v_min)] = v_min
        temp_arr[np.where(temp_arr>v_max)] = v_max
        temp_arr = (temp_arr-v_min)/(v_max-v_min)
        list_list_values.append(temp_arr)

        for idx, xy in enumerate(pc_os64['img_idx']):
            temp_img[xy[0], xy[1], channel_idx] = temp_arr[idx]

    if is_flip:
        temp_img = np.flip(np.flip(temp_img, 0), 1).copy()

    return temp_img

def get_filtered_point_cloud_from_plain_text(p_frame, is_with_list_infos=False):
    # path_pcd = os.path.join(cnf_ui.BASE_DIR, 'data', 'example', f'pc_{p_frame.str_time}.pcd')
    path_pcd = p_frame.dict_lidar['pc']
    
    pc_os64 = get_pc_os64_with_path(path_pcd)

    plain_text = p_frame.plainTextEditLabels.toPlainText()
    list_dict_bbox = get_list_dict_by_processing_plain_text(plain_text)

    if len(list_dict_bbox) == 0:
        p_frame.addLogs('no bboxes!')
        return None

    if p_frame.spinBoxIndex_0.value() >= len(list_dict_bbox):
        p_frame.addLogs('no bboxes in the index!')
        return None

    dict_bbox = list_dict_bbox[p_frame.spinBoxIndex_0.value()]
    list_infos = [dict_bbox['x'], dict_bbox['y'], \
        dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l']]

    # Move to the bbox center
    list_pc_values = pc_os64['values'].tolist()
    pc_os64['values'] = np.array(list(map(lambda point: \
        [point[0]-list_infos[0], point[1]-list_infos[1], \
                        point[2], point[3]], list_pc_values)))

    # Limit the rough roi
    pc_os64 = filter_pc_os64_with_roi(pc_os64, [-5, 5, -5, 5])

    # give rotation (-theta)
    azi_rad = -list_infos[2]/180.*np.pi
    c_y = np.cos(azi_rad)
    s_y = np.sin(azi_rad)
    R_yaw = np.array([[c_y, -s_y, 0.],[s_y, c_y, 0.],[0., 0., 1.]])
    pc_xyz = pc_os64['values'][:,:3].copy()
    num_points, _ = pc_os64['values'].shape
    
    for i in range(num_points):
        point_temp = pc_xyz[i,:]
        point_temp = np.reshape(point_temp, (3,1))
        point_processed = np.dot(R_yaw, point_temp)
        point_processed = np.reshape(point_processed, (3,))
        pc_xyz[i,:] = point_processed

    pc_os64['values'][:,:3] = pc_xyz

    # Limit the roi again
    pc_os64 = filter_pc_os64_with_roi(pc_os64, \
        [-list_infos[3], list_infos[3], -list_infos[4], list_infos[4]])

    if is_with_list_infos:
        return pc_os64, list_infos
    else:
        return pc_os64

def get_pixel_index_from_m_coordinate(v0, v1, type='yz'):
    if type == 'yz':
        range_v0 = cnf_ui.RANGE_Y_FRONT
        range_v1 = cnf_ui.RANGE_Z_FRONT
        m_per_pix = cnf_ui.M_PER_PIX_YZ
        img_size = cnf_ui.IMG_SIZE_YZ
    elif type == 'xz':
        range_v0 = cnf_ui.RANGE_X_FRONT
        range_v1 = cnf_ui.RANGE_Z_FRONT
        m_per_pix = cnf_ui.M_PER_PIX_XZ
        img_size = cnf_ui.IMG_SIZE_XZ

    x = int(np.round((v0-range_v0[0])/m_per_pix))
    y = int(np.round(img_size[0]-(v1-range_v1[0])/m_per_pix))

    if (x >= 0) & (x < img_size[0]) & (y >= 0) & (y < img_size[0]):
        return x, y
    else:
        return None, None

def get_front_beside_image_from_point_cloud(pc_os64, radius=1, color=(0,0,0)):
    # pc -> front img
    pc_xyz = pc_os64['values'].copy()
    list_x = pc_xyz[:,0]
    list_y = pc_xyz[:,1]
    list_z = pc_xyz[:,2]
    # list_i = pc_xyz[:,3]

    # YZ: front img
    img_h, img_w = cnf_ui.IMG_SIZE_YZ
    img_bev_f = np.full((img_h,img_w,3), 255, dtype=np.uint8)
    for y, z in zip(list_y, list_z):
        pix_x, pix_y = get_pixel_index_from_m_coordinate(y, z, 'yz')
        if pix_x:
            img_bev_f = cv2.circle(img_bev_f, (pix_x, pix_y), radius, color, thickness=-1)
    
    img_h, img_w = cnf_ui.IMG_SIZE_XZ
    img_bev_b = np.full((img_h,img_w,3), 255, dtype=np.uint8)
    for x, z in zip(list_x, list_z):
        pix_x, pix_y = get_pixel_index_from_m_coordinate(x, z, 'xz')
        if pix_x:
            img_bev_b = cv2.circle(img_bev_b, (pix_x, pix_y), radius, color, thickness=-1)
    
    return img_bev_f, img_bev_b

def get_o3d_point_cloud(arr_pc, color=None):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(arr_pc[:,:3])
    
    if color is not None:
        len_points = len(arr_pc[:,:3])
        pcd.colors = o3d.utility.Vector3dVector([color for _ in range(len_points)])

    return pcd

def filter_pc_with_roi_in_xyz(arr_pc, list_roi_xyz):
    x_min, x_max, y_min, y_max, z_min, z_max = list_roi_xyz
    list_pc_values = arr_pc.tolist()
    pc_filtered = np.array(list(filter(lambda point: \
        (point[0] > x_min) and (point[0] < x_max) and \
        (point[1] > y_min) and (point[1] < y_max) and \
        (point[2] > z_min) and (point[2] < z_max), list_pc_values)))
    
    return pc_filtered

def get_o3d_line_set_from_tuple_bbox(tuple_bbox, is_with_arrow=True, length_arrow=1.0, length_tips=0.4, cfg=None):
    name_cls, idx_cls, list_values, _ = tuple_bbox
    x, y, z, theta, l, w, h = list_values

    points = [
        [l/2, w/2, h/2],            # 0
        [l/2, w/2, -h/2],           # 1
        [l/2, -w/2, h/2],           # 2
        [l/2, -w/2, -h/2],          # 3
        [-l/2, w/2, h/2],           # 4
        [-l/2, w/2, -h/2],          # 5
        [-l/2, -w/2, h/2],          # 6
        [-l/2, -w/2, -h/2],         # 7
    ]

    if is_with_arrow:
        points.extend([
            [0, 0, 0],                  # 8
            [l/2+length_arrow, 0, 0],   # 9
            [l/2+length_arrow-length_tips, length_tips, 0],     # 10
            [l/2+length_arrow-length_tips, -length_tips, 0],    # 11
        ])

    ### Rotation ###
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    mat_rot = np.array([
        [cos_th, -sin_th, 0],
        [sin_th, cos_th, 0],
        [0, 0, 1]
    ])
    points = list(map(lambda point: mat_rot.dot(np.array(point).reshape((3,1))).reshape(1,3).tolist()[0],points))
    ### Rotation ###

    ### Translation ###
    points = list(map(lambda point: [point[0]+x, point[1]+y, point[2]+z], points))


    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], \
        [1, 5], [2, 3], [2, 6], [3, 7], \
        [4, 5], [4, 6], [5, 7], [6, 7], \
    ]

    if is_with_arrow:
        lines.extend([
            [8, 9], [9, 10], [9, 11], \
        ])

    if not (cfg is None):
        color = cfg.DATASET.CLASS_RGB[name_cls]
    else:
        color = cnf.DIC_CLS_RGB[name_cls]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def get_points_power_from_cube_bev(cube_bev, bin_size, power_multiplier=10.0, power_offset=-10.0, is_flip=False, roi_point=None):
    _, bin_y, bin_x = bin_size # [m/bin]
    len_y, len_x = cube_bev.shape
    
    half_y = len_y/2.

    arr_y = (np.arange(len_y, dtype=float)-half_y+0.5)*bin_y
    arr_x = (np.arange(len_x, dtype=float)+0.5)*bin_x

    list_points = []
    for j in range(len_y):
        for i in range(len_x):
            # if cube_bev[j,i] < 1e-7:
            #     continue

            power_to_vis = power_multiplier*(cube_bev[j,i]+power_offset)
            if is_flip:
                power_to_vis = -power_to_vis
            list_points.append([arr_x[i], arr_y[j], power_to_vis])

    return np.array(list_points)
    
def get_list_bboxes_tuples_from_inference(arr_rpn, bin_size, arr_reg, thr_rpn=0.5, is_vis=False, cfg=None, cls=None):
    bin_z, bin_y, bin_x = bin_size # [m/bin]
    len_y, len_x = arr_rpn.shape
    
    half_y = len_y/2.

    arr_y = (np.arange(len_y, dtype=float)-half_y+0.5)*bin_y
    arr_x = (np.arange(len_x, dtype=float)+0.5)*bin_x
    
    list_y, list_x = np.where(arr_rpn>thr_rpn)
    
    if is_vis:
        arr_vis = np.zeros((len_y,len_x))
        for j, i in zip(list_y, list_x):
            arr_vis[j,i] = 1
        cv2.imshow('conf', arr_vis)
        cv2.waitKey(0)
    
    list_tuples = []
    idx_obj = 0
    for j, i in zip(list_y, list_x):
        values = arr_reg[:,j,i]
        # print(values)

        z_cen = cfg.VIS.Z_CENTER_DIC[cls]
        y_cen = arr_y[j]
        # y_cen = values[1]
        x_cen = arr_x[i]
        # x_cen = values[0]
        
        # print(x_cen, y_cen)

        z_len = cfg.VIS.Z_HEIGHT_DIC[cls]
        y_len = values[3]
        x_len = values[2]

        th_rad = values[4]
        th_deg = th_rad#*180./np.pi
        
        id_cls = cfg.DATASET.CLASS_ID[cls]
        
        temp_tuple = (cls, cfg.DATASET.CLASS_ID[cls], [x_cen, y_cen, z_cen, th_deg, x_len, y_len, z_len], idx_obj)
        list_tuples.append(temp_tuple)
        idx_obj += 1

    return list_tuples

def get_o3d_line_set_from_list_infos(list_infos, color = [0., 0., 0.], is_with_arrow=True, length_arrow=1.0, length_tips=0.4):
    x, y, z, azi_deg, l_2, w_2, h_2 = list_infos
    theta = azi_deg*np.pi/180.
    l = l_2*2.
    w = w_2*2.
    h = h_2*2

    points = [
        [l/2, w/2, h/2],            # 0
        [l/2, w/2, -h/2],           # 1
        [l/2, -w/2, h/2],           # 2
        [l/2, -w/2, -h/2],          # 3
        [-l/2, w/2, h/2],           # 4
        [-l/2, w/2, -h/2],          # 5
        [-l/2, -w/2, h/2],          # 6
        [-l/2, -w/2, -h/2],         # 7
    ]

    if is_with_arrow:
        points.extend([
            [0, 0, 0],                  # 8
            [l/2+length_arrow, 0, 0],   # 9
            [l/2+length_arrow-length_tips, length_tips, 0],     # 10
            [l/2+length_arrow-length_tips, -length_tips, 0],    # 11
        ])

    ### Rotation ###
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    mat_rot = np.array([
        [cos_th, -sin_th, 0],
        [sin_th, cos_th, 0],
        [0, 0, 1]
    ])
    points = list(map(lambda point: mat_rot.dot(np.array(point).reshape((3,1))).reshape(1,3).tolist()[0],points))

    ### Translation ###
    points = list(map(lambda point: [point[0]+x, point[1]+y, point[2]+z], points))


    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], \
        [1, 5], [2, 3], [2, 6], [3, 7], \
        [4, 5], [4, 6], [5, 7], [6, 7], \
    ]

    if is_with_arrow:
        lines.extend([
            [8, 9], [9, 10], [9, 11], \
        ])

    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

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
