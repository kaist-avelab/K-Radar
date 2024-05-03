'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from easydict import EasyDict

### Configuration ###
DICT_CFG = {
    'DIR': {
        'LIST_DIR': ['/media/donghee/HDD_3/K-Radar/radar_bin_lidar_bag_files/generated_files'],
        'DIR_FLIT_SRT': '/media/donghee/HDD_3/K-Radar/dir_sp_filtered',
        'DIR_REVISED_LABEL': '/media/donghee/HDD_3/K-Radar/kradar_revised_label',
        'DIR_SAVE_RENDER_LIDAR': '/media/donghee/HDD_3/K-Radar/dir_rendered_lpc'
    },
    'LIDAR': {
        'ROI': {
            'x': [0,80],
            'y': [-40,40],
            'z': [-2,6],
        },
        'CALIB_Z': 0.7,
    },
    'RENDER': {
        'x': [0,0.1,80],
        'y': [-40,0.1,40],
        'hue': 'z',
        'val': 'intensity',
        'z_roi': [-2,6],
        'intensity_roi': [0,2048],
        'dilation': 11,
    },
}
### Configuration ###

class PointCloudPcd():
    def __init__(self, path_pcd:str, len_header:int=11, ego_offset:float=1e-3)->object:
        f = open(path_pcd, 'r')
        lines = f.readlines()
        f.close()
        self.path_pcd = path_pcd

        list_header = lines[:len_header]
        list_values = lines[len_header:]
        list_values = list(map(lambda x: x.split(' '), list_values))
        values = np.array(list_values, dtype=np.float32)
        values = values[ # delete (0,0)
            np.where(
                (values[:,0]<-ego_offset) | (values[:,0]>ego_offset) |  # x
                (values[:,1]<-ego_offset) | (values[:,1]>ego_offset)    # y
            )]
        self.values = values
        self.list_attr = (list_header[2].rstrip('\n')).split(' ')[1:]
        self.is_calibrated = False
        self.is_roi_filtered = False

    def __repr__(self)->str:
        str_repr = f'total {len(self.values)}x{len(self.list_attr)} points, fields = {self.list_attr}'
        if self.is_calibrated:
            str_repr += ', calibrated'
        if self.is_roi_filtered:
            str_repr += ', roi filtered'
        return str_repr
    
    @property
    def points(self): # x, y, z
        return self.values[:,:3]
    
    @property
    def points_w_attr(self):
        return self.values

    def _get_o3d_pcd(self)->o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd

    def _get_bev_pcd(self, dict_render)->np.array:
        x_min, x_bin, x_max = dict_render['x']
        y_min, y_bin, y_max = dict_render['y']

        hue_type = dict_render['hue']
        val_type = dict_render['val']

        pts_w_attr = (self.points_w_attr.copy()).tolist()
        pts_w_attr = np.array(sorted(pts_w_attr,key=lambda x: x[2])) # sort via z

        arr_x = np.linspace(x_min, x_max-x_bin, num=int((x_max-x_min)/x_bin)) + x_bin/2.
        arr_y = np.linspace(y_min, y_max-y_bin, num=int((y_max-y_min)/y_bin)) + y_bin/2.
        
        xy_mesh_grid_hsv = np.full((len(arr_x), len(arr_y), 3), 0, dtype=np.int64)
        x_idx = np.clip(((pts_w_attr[:,0]-x_min)/x_bin+x_bin/2.).astype(np.int64),0,len(arr_x)-1)
        y_idx = np.clip(((pts_w_attr[:,1]-y_min)/y_bin+y_bin/2.).astype(np.int64),0,len(arr_y)-1)

        hue_min, hue_max = dict_render[f'{hue_type}_roi']
        hue_val = np.clip((pts_w_attr[:,self.list_attr.index(hue_type)]-hue_min)/(hue_max-hue_min),0.1,0.9)

        val_min, val_max = dict_render[f'{val_type}_roi']
        val_val = np.clip((pts_w_attr[:,self.list_attr.index(val_type)]-val_min)/(val_max-val_min),0.5,0.9)

        xy_mesh_grid_hsv[x_idx,y_idx,0] = (hue_val*127.).astype(np.int64)
        xy_mesh_grid_hsv[x_idx,y_idx,1] = 255 # Saturation
        xy_mesh_grid_hsv[x_idx,y_idx,2] = (val_val*255.).astype(np.int64)

        xy_mesh_grid_rgb_temp = cv2.cvtColor(xy_mesh_grid_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        dilation = dict_render['dilation']
        xy_mesh_grid_rgb_temp = cv2.dilate(xy_mesh_grid_rgb_temp, kernel=(dilation,dilation))

        xy_mesh_grid_rgb = np.full_like(xy_mesh_grid_rgb_temp, fill_value=255, dtype=np.uint8)
        x_ind_valid, y_ind_valid = np.where(np.sum(xy_mesh_grid_rgb_temp, axis=2)>0)
        xy_mesh_grid_rgb[x_ind_valid,y_ind_valid,:] = xy_mesh_grid_rgb_temp[x_ind_valid,y_ind_valid,:]

        xy_mesh_grid_rgb = np.flip(xy_mesh_grid_rgb, axis=(0,1))

        return xy_mesh_grid_rgb

    def calib_xyz(self, list_calib_xyz:list):
        arr_calib_xyz = np.array(list_calib_xyz, dtype=self.values.dtype).reshape(1,3)
        arr_calib_xyz = arr_calib_xyz.repeat(repeats=len(self.values), axis=0)
        self.values[:,:3] += arr_calib_xyz
        self.is_calibrated=True

    def roi_filter(self, dict_roi:dict):
        '''
        dict_roi
            key: 'attr', value: [attr_min, attr_max]
        e.g., {'x': [0, 100]}
        '''
        values = self.values.copy()
        for temp_key, v in dict_roi.items():
            if not (temp_key in self.list_attr):
                print(f'* {temp_key} is not in attr')
                continue
            v_min, v_max = v
            idx = self.list_attr.index(temp_key)
            values = values[
                np.where(
                    (values[:,idx]>v_min) & (values[:,idx]<v_max)
                )]
        self.values = values
        self.is_roi_filtered=True

    def render_in_o3d(self):
        o3d.visualization.draw_geometries([self._get_o3d_pcd()])

    def render_in_bev(self, dict_render:dict):
        img_bev = self._get_bev_pcd(dict_render)
        cv2.imshow('LiDAR PCD (in BEV)', img_bev)
        cv2.waitKey(0)

class SamplesPointCloudPcd():
    def __init__(self, cfg:dict=None)->object:
        self.cfg = cfg
        self.list_path_lidar = []
        for dir_seq in cfg.DIR.LIST_DIR:
            list_seq = os.listdir(dir_seq)
            for seq in list_seq:
                seq_lidar_paths = sorted(glob(osp.join(dir_seq, seq, 'os2-64', 'os2-64_*.pcd')))
                self.list_path_lidar.extend(seq_lidar_paths)
        
        self.roi = cfg.LIDAR.get('ROI', None)
        self.render = cfg.get('RENDER', None)

    def __getitem__(self, idx:int)->PointCloudPcd:
        path_pcd = self.list_path_lidar[idx]
        pcd = PointCloudPcd(path_pcd)
        path_calib = osp.sep+osp.join(*path_pcd.split('/')[:-2],\
                            'info_calib', 'calib_radar_lidar.txt')
        if osp.exists(path_calib):
            list_calib_xyz = self._get_calib_info(path_calib)
            pcd.calib_xyz(list_calib_xyz)
        if self.roi is not None:
            pcd.roi_filter(self.roi)
        return pcd

    def __len__(self)->int:
        return len(self.list_path_lidar)
    
    def _get_calib_info(self, path_calib:str)->list:
        f = open(path_calib)
        lines = f.readlines()
        list_calib_val = list(map(lambda x: float(x), lines[1].split(',')))[1:]
        list_calib_val.append(self.cfg.LIDAR.CALIB_Z)
        f.close()
        return list_calib_val

    def save_bev_pcd(self):
        path_render_lidar = self.cfg.DIR.DIR_SAVE_RENDER_LIDAR
        for i in range(58):
            os.makedirs(osp.join(path_render_lidar, f'{i+1}'), exist_ok=True)
        for idx_item in tqdm(range(self.__len__())):
            temp_pcd = self.__getitem__(idx_item)
            path_pcd = temp_pcd.path_pcd
            rendered_img = temp_pcd._get_bev_pcd(self.render)
            path_split = path_pcd.split('/')
            seq = path_split[-3]
            file_name = path_split[-1].split('.')[0]+'.png'
            cv2.imwrite(osp.join(path_render_lidar, seq, file_name), rendered_img)

if __name__ == '__main__':
    samples_pcd = SamplesPointCloudPcd(cfg=EasyDict(DICT_CFG))
    
    # pcd = samples_pcd[0]
    # print(pcd)
    # pcd.render_in_o3d()
    # pcd.render_in_bev(samples_pcd.render)

    samples_pcd.save_bev_pcd()
