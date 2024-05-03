'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

from tqdm import tqdm
from scipy import io, ndimage
from easydict import EasyDict

try:
    from .radar_preprocessor_utils import *
except:
    import sys
    sys.path.append(osp.dirname(osp.abspath(__file__)))
    from radar_preprocessor_utils import *

CFG_DICT = {
    'RDR_TESSERACT': {
        'PATH_RDAE_VALUE': {
            'RAE': './resources/info_arr.mat',
            'DOPPLER': './resources/arr_doppler.mat',
        },
        'IS_REVERSE_AE': True,
        'IS_CONSIDER_ROI_POLAR': False,
        'ROI_POLAR': { # from old cfg (reversed here)
            'range': [0, 100],
            'azimuth': [-52, 51],
            'elevation': [-4, 15],
        },
        'IS_CONSIDER_ROI_CARTESIAN': True,
        'ROI_CARTESIAN': {
            'x': [1, 110], # [1, 120] for vis, [1, 72] for default
            'y': [-20, 20],
            'z': [-2, 2], # [-2, 2] for vis, [-2, 6] for default
        },
        'CA_CFAR': {
            'FALSE_ALARM_RATE': 0.005, # 0.0005
            'GUARD_CELL_RAE': [4, 2, 2],
            'TRAIN_CELL_RAE': [8, 4, 4],
        },
        'OS_CFAR': {
            'RATE': 0.05,
            'PADDING_HALF_RA': [2,1],
        },
        'CA_CFAR_RA': {
            'FALSE_ALARM_RATE': 0.05, # 0.0005
            'GUARD_CELL_RA': [4, 2],
            'TRAIN_CELL_RA': [8, 4],
            'VAL_Z': 0.5,
        }
    },
    'LIDAR': {
        'IS_CONSIDER_ROI': True,
        'ROI': {
            'x': [0, 80], # [1, 120] for vis, [1, 80] for default
            'y': [-40, 40],
            'z': [-2, 6],
        },
        'CALIB_Z': 0.7,
    },
    'RENDER_RDR_BEV': {
        'PATH_SAVE_PLOT': {
            'BUFFER': './resources/imgs/img_tes_ra.png',
            'PLOT': './resources/imgs/plot_tes_ra.png',
        },
        'ROI_X': [0, 0.2, 80], 
        'ROI_Y': [-40, 0.2, 40],
    },
    'RENDER_RPC': {
        'STEP': 0,
        'RGB_STEP': {
            'STEP0': [0.,0.,0.],
            'STEP1': [0.,0.,0.],
            'STEP2': [0.,0.,1.],
            'STEP3': [0.,0.,1.],
        },
        'IMG_SIZE_BEV_XY': [400, 400],
        'DILATION': 21,
    },
    'RENDER_LDR_BEV': {
        'ROI_X': [0,0.1,80],
        'ROI_Y': [-40,0.1,40],
        'HUE': 'z',
        'VAL': 'intensity',
        'z_ROI': [-2,6],
        'intensity_ROI': [0,2048],
        'DILATION': 21,
    },
    'RENDER_LPC': {

    },
    'RENDER_OBJ': {
        'DICT_CLS_NAME_TO_BGR': {
            'Sedan': [0,50,255],
            'Bus or Truck': [255,0,255],
            'Motorcycle': [0,0,255],
            'Bicycle': [50,0,255],
            'Pedestrian': [255,0,0],
            'Pedestrian Group': [255,0,100],
            'Label': [0,0,0],
        },
        'DICT_CLS_NAME_TO_RGB': {
            'Sedan': [1, 0.2, 0],
            'Bus or Truck': [1, 0, 1],
            'Motorcycle': [1, 0, 0],
            'Bicycle': [1, 0, 0.2],
            'Pedestrian': [0, 0, 1],
            'Pedestrian Group': [0.4, 0, 1],
            'Label': [0, 0, 0],
        },
        'LINEWIDTH': 2,
        'ALPHA': 0.5,
        'RADIUS': 0,
        'CONF_TEXT_SIZE': 0.8,
    },
    'RENDER_LOAD': {
        'ROI_X': [0,0.2,80],
        'ROI_Y': [-40,0.2,40],
        'DILATION': 21,
    }
}

class PointCloudPcdWithSampled():
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
        x_min, x_bin, x_max = dict_render['ROI_X']
        y_min, y_bin, y_max = dict_render['ROI_Y']

        hue_type = dict_render['HUE']
        val_type = dict_render['VAL']

        pts_w_attr = (self.points_w_attr.copy()).tolist()
        pts_w_attr = np.array(sorted(pts_w_attr,key=lambda x: x[2])) # sort via z

        arr_x = np.linspace(x_min, x_max-x_bin, num=int((x_max-x_min)/x_bin)) + x_bin/2.
        arr_y = np.linspace(y_min, y_max-y_bin, num=int((y_max-y_min)/y_bin)) + y_bin/2.
        
        xy_mesh_grid_hsv = np.full((len(arr_x), len(arr_y), 3), 0, dtype=np.int64)
        x_idx = np.clip(((pts_w_attr[:,0]-x_min)/x_bin+x_bin/2.).astype(np.int64),0,len(arr_x)-1)
        y_idx = np.clip(((pts_w_attr[:,1]-y_min)/y_bin+y_bin/2.).astype(np.int64),0,len(arr_y)-1)

        hue_min, hue_max = dict_render[f'{hue_type}_ROI']
        hue_val = np.clip((pts_w_attr[:,self.list_attr.index(hue_type)]-hue_min)/(hue_max-hue_min),0.1,0.9)

        val_min, val_max = dict_render[f'{val_type}_ROI']
        val_val = np.clip((pts_w_attr[:,self.list_attr.index(val_type)]-val_min)/(val_max-val_min),0.5,0.9)

        xy_mesh_grid_hsv[x_idx,y_idx,0] = (hue_val*127.).astype(np.int64)
        xy_mesh_grid_hsv[x_idx,y_idx,1] = 255 # Saturation
        xy_mesh_grid_hsv[x_idx,y_idx,2] = (val_val*255.).astype(np.int64)

        xy_mesh_grid_rgb_temp = cv2.cvtColor(xy_mesh_grid_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        dilation = dict_render['DILATION']
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

    def sampling_per_azimuth(self, sampling_rate:float=0.5):
        print(f'* sampling per azimuth in {sampling_rate}')
        list_sampled_per_ring = []
        for idx_ring in np.unique(self.values[:,6]):
            val_temp = self.values[np.where(self.values[:,6]==idx_ring)[0],:]
            n_points, _ = val_temp.shape
            arr_idx = np.random.choice(n_points, int(n_points*sampling_rate), replace=False)
            sampled = val_temp[arr_idx,:]
            list_sampled_per_ring.append(sampled)
            # print(val_temp.shape)
            # print(sampled.shape)
        self.values = np.concatenate(list_sampled_per_ring, axis=0)

class RadarPreprocessor():
    '''
    Preprocessor for 4D Radar (ICCV2023)
    '''
    def __init__(self, cfg:dict=None)->object:
        self.cfg = cfg
        self.cfg_rdr = self.cfg.RDR_TESSERACT
        self.cfg_ldr = self.cfg.LIDAR
        self.cfg_render_rdr_bev = self.cfg.RENDER_RDR_BEV
        self.cfg_render_ldr_bev = self.cfg.RENDER_LDR_BEV
        self.cfg_render_rpc = self.cfg.RENDER_RPC
        self.cfg_render_lpc = self.cfg.RENDER_LPC
        
        ### Tesseract ###
        self._is_reverse_ae = self.cfg_rdr.IS_REVERSE_AE
        self.arr_range, self.arr_doppler, self.arr_azimuth,\
                        self.arr_elevation = self._load_rdae_value()
        self._is_consider_roi_polar = self.cfg_rdr.get('IS_CONSIDER_ROI_POLAR', False)
        if self._is_consider_roi_polar: # ROI indexing in dense tensor
            self._roi_polar_idx, self._roi_polar = self._get_roi_polar(self.cfg_rdr.ROI_POLAR)
        self._is_consider_roi_cartesian_rdr = self.cfg_rdr.get('IS_CONSIDER_ROI_CARTESIAN', False)
        if self._is_consider_roi_cartesian_rdr:
            self._roi_cartesian_rdr = self.cfg_rdr.ROI_CARTESIAN
        ### Tesseract ###

        ### Object Renderer ###
        self.object_renderer = ObjectRenderer(cfg.RENDER_OBJ)
        self.list_calib_xyz = None # for LiDAR & Label
        ### Object Renderer ###
    
    def _load_rdae_value(self)->list:
        dict_path_rdae = self.cfg_rdr.PATH_RDAE_VALUE
        rae_values = io.loadmat(dict_path_rdae.RAE)
        arr_range = (rae_values['arrRange']).reshape(-1,)
        deg2rad = np.pi/180.
        arr_azimuth = (rae_values['arrAzimuth']*deg2rad).reshape(-1,)
        arr_elevation = (rae_values['arrElevation']*deg2rad).reshape(-1,)
        arr_doppler = (io.loadmat(dict_path_rdae.DOPPLER)['arr_doppler']).reshape(-1,)
        if self._is_reverse_ae:
            arr_azimuth = np.flip(-arr_azimuth)
            arr_elevation = np.flip(-arr_elevation)
        return arr_range, arr_doppler, arr_azimuth, arr_elevation

    def _load_rdr_tesseract(self, path_rdr_tesseract:str, is_doppler_separated:bool=False)->any:
        arr_tesseract = io.loadmat(path_rdr_tesseract)['arrDREA']
        arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2)) # in DRAE
        if self._is_reverse_ae:
            arr_tesseract = np.flip(np.flip(arr_tesseract, axis=2), axis=3)
        if self._is_consider_roi_polar: # filtering in polar w/ indexing
            idx_r_0, idx_r_1, idx_a_0, idx_a_1, idx_e_0, idx_e_1 = self._roi_polar_idx
            arr_tesseract = arr_tesseract[:,idx_r_0:idx_r_1+1,\
                                            idx_a_0:idx_a_1+1,idx_e_0:idx_e_1+1]
        if is_doppler_separated:
            arr_tesseract_t = np.transpose(arr_tesseract, [1,2,3,0])
            sum_doppler = np.sum(arr_tesseract_t, axis=3, keepdims=True)
            arr_tesseract_t_norm = arr_tesseract_t / sum_doppler
            arr_tesseract_t_expectation = np.sum(arr_tesseract_t_norm * np.array([i for i in range(64)]), axis=3)
            arr_tesseract_t_expectation_idx = np.clip(arr_tesseract_t_expectation.astype(np.uint8), 0, 63)
            cube_doppler = self.arr_doppler[arr_tesseract_t_expectation_idx] # R, A, E
            cube_power = np.mean(arr_tesseract, axis=0)
            return cube_power, cube_doppler # R, A, E
        else:
            return arr_tesseract # D,R,A,E
    
    def _load_ldr_pcd(self, path_pcd:str, sampling_per_azimuth:float=None)->PointCloudPcdWithSampled:
        pcd = PointCloudPcdWithSampled(path_pcd)
        if not (sampling_per_azimuth is None):
            pcd.sampling_per_azimuth(sampling_per_azimuth)
        if not (self.list_calib_xyz is None):
            pcd.calib_xyz(self.list_calib_xyz)
        if self.cfg_ldr.get('IS_CONSIDER_ROI', False):
            pcd.roi_filter(self.cfg_ldr.ROI)
        return pcd
    
    def _get_roi_polar(self, dict_roi_polar:dict)->list:
        roi_polar_idx = [0, len(self.arr_range)-1, \
            0, len(self.arr_azimuth)-1, 0, len(self.arr_elevation)-1]
        idx_attr = 0
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi

        def get_arr_in_roi(arr, min_max):
            min_val, max_val = min_max
            idx_min = np.argmin(abs(arr-min_val))
            idx_max = np.argmin(abs(arr-max_val))
            return arr[idx_min:idx_max+1], idx_min, idx_max
        
        for k, v in dict_roi_polar.items():
            if v is not None:
                min_max = (np.array(v)*deg2rad).tolist() if idx_attr > 0 else v
                arr_roi, idx_min, idx_max = get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                roi_polar_idx[idx_attr*2] = idx_min
                roi_polar_idx[idx_attr*2+1] = idx_max
                
                v_new = [arr_roi[0], arr_roi[-1]]
                v_new =  (np.array(v_new)*rad2deg) if idx_attr > 0 else v_new
                dict_roi_polar[k] = v_new
            idx_attr += 1
        return roi_polar_idx, dict_roi_polar

    def _get_rendered_bev_img(self, arr_tesseract:np.array)->np.array:
        dict_path_save_plot = self.cfg_render_rdr_bev.PATH_SAVE_PLOT
        rdr_bev = np.mean(np.mean(arr_tesseract, axis=0), axis=2)
        arr_0, arr_1 = np.meshgrid(self.arr_azimuth, self.arr_range)
        height, width = np.shape(rdr_bev)
        plt.clf()
        plt.cla()
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.savefig(dict_path_save_plot.BUFFER, bbox_inches='tight', pad_inches=0, dpi=300)
        temp_img = cv2.imread(dict_path_save_plot.BUFFER)
        temp_row, temp_col, _ = temp_img.shape
        if not (temp_row == height and temp_col == width):
            temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dict_path_save_plot.BUFFER, temp_img_new)
        plt.close()
        if not (dict_path_save_plot.get('PLOT', None) is None):
            plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
            plt.colorbar()
            plt.savefig(dict_path_save_plot.PLOT, dpi=300)
            plt.close()
        ra = cv2.imread(dict_path_save_plot.BUFFER)
        ra = np.flip(ra, axis=0) # Polar to Cartesian (Should flip image)
        arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra,\
                self.arr_range, self.arr_azimuth,\
                roi_x = self.cfg_render_rdr_bev.ROI_X, roi_y = self.cfg_render_rdr_bev.ROI_Y, is_in_deg=False)
        arr_yx = arr_yx.transpose((1,0,2))
        arr_yx = np.flip(arr_yx, axis=0)
        return arr_yx
    
    def _get_calib_info(self, path_calib:str)->list:
        f = open(path_calib)
        lines = f.readlines()
        list_calib_val = list(map(lambda x: float(x), lines[1].split(',')))[1:]
        list_calib_val.append(self.cfg.LIDAR.CALIB_Z)
        f.close()
        return list_calib_val
    
    def _get_processed_3D_rdr_from_path_tesseract(self, path_tesseract:str)->any:
        '''
        To show difference between 4D Radar & 3D Radar
        '''
        cube_pw, cube_dop = self._load_rdr_tesseract(path_tesseract, is_doppler_separated=True)
        matrix_pw = np.mean(cube_pw, axis=2)
        matrix_dop = np.mean(cube_dop, axis=2)
        ind_r, ind_a = self._get_ca_cfar_idx_from_ra_matrix(matrix_pw)

        val_r = self.arr_range[ind_r]       # N,
        val_a = self.arr_azimuth[ind_a]     # N,

        val_x = val_r*np.cos(val_a)
        val_y = val_r*np.sin(val_a)
        val_z = np.full_like(val_x, self.cfg_rdr.CA_CFAR_RA.VAL_Z, dtype=val_x.dtype)
        val_pw = matrix_pw[ind_r, ind_a]
        val_dop = matrix_dop[ind_r, ind_a]

        pts_ca_cfar = np.column_stack((val_x,val_y,val_z,val_pw,val_dop)) # X, Y, Z(fixed), pw, dop
        
        if self._is_consider_roi_cartesian_rdr:
            dict_roi = self._roi_cartesian_rdr
            x_min, x_max = dict_roi['x']
            y_min, y_max = dict_roi['y']
            ind_valid_pts = np.where(
                (pts_ca_cfar[:,0]>x_min) & (pts_ca_cfar[:,0]<x_max) &
                (pts_ca_cfar[:,1]>y_min) & (pts_ca_cfar[:,1]<y_max) 
            )[0]
            pts_ca_cfar = pts_ca_cfar[ind_valid_pts,:]
            ind_r = ind_r[ind_valid_pts]
            ind_a = ind_a[ind_valid_pts]
        
        return pts_ca_cfar
    
    def _get_processed_rdr_from_path_tesseract(self, path_tesseract:str, step:int=3)->any:
        '''
        Step 1. CA-CFAR
        '''
        cube_pw, cube_dop = self._load_rdr_tesseract(path_tesseract, is_doppler_separated=True)
        ind_r, ind_a, ind_e = self._get_ca_cfar_idx_from_cube(cube_pw)

        val_r = self.arr_range[ind_r]       # N,
        val_a = self.arr_azimuth[ind_a]     # N,
        val_e = self.arr_elevation[ind_e]   # N,

        val_x = val_r*np.cos(val_e)*np.cos(val_a)
        val_y = val_r*np.cos(val_e)*np.sin(val_a)
        val_z = val_r*np.sin(val_e)
        val_pw = cube_pw[ind_r, ind_a, ind_e]
        val_dop = cube_dop[ind_r, ind_a, ind_e]

        pts_ca_cfar = np.column_stack((val_x,val_y,val_z,val_pw,val_dop))
        
        if self._is_consider_roi_cartesian_rdr:
            dict_roi = self._roi_cartesian_rdr
            x_min, x_max = dict_roi['x']
            y_min, y_max = dict_roi['y']
            z_min, z_max = dict_roi['z']
            ind_valid_pts = np.where(
                (pts_ca_cfar[:,0]>x_min) & (pts_ca_cfar[:,0]<x_max) &
                (pts_ca_cfar[:,1]>y_min) & (pts_ca_cfar[:,1]<y_max) &
                (pts_ca_cfar[:,2]>z_min) & (pts_ca_cfar[:,2]<z_max)
            )[0]
            pts_ca_cfar = pts_ca_cfar[ind_valid_pts,:]
            ind_r = ind_r[ind_valid_pts]
            ind_a = ind_a[ind_valid_pts]
            ind_e = ind_e[ind_valid_pts]

        if step == 1: # CA-CFAR w/ high false alarm rate
            return pts_ca_cfar # N,5
        
        '''
        Step 2. OS-CFAR along azimuth
        Weighted (along E) RAE map: pw*(20-2*abs(z)) weight is largest at z=0 (due to high pw at wheels)
        '''
        rae_map = np.zeros((len(self.arr_range), len(self.arr_azimuth), len(self.arr_elevation)), dtype=val_r.dtype)
        rae_map[ind_r,ind_a,ind_e] = pts_ca_cfar[:,3]*(20.-2.*np.abs(pts_ca_cfar[:,2]))
        ra_map = np.sum(rae_map, axis=2)


        cfg_os_cfar = self.cfg_rdr.OS_CFAR

        temp_thr_along_range = np.quantile(ra_map, 1.-cfg_os_cfar.RATE, axis=1)
        temp_thr_along_range[np.where(temp_thr_along_range==0.)] = sys.float_info.max
        temp_thr_along_range = temp_thr_along_range.reshape(-1,1).repeat(len(self.arr_azimuth), axis=-1)

        bool_ra_map_ca_cfar = np.full_like(ra_map, fill_value=False, dtype=bool)
        bool_ra_map_ca_cfar[ind_r,ind_a] = True
        bool_ra_map_os_cfar = ra_map > temp_thr_along_range
        # print(np.where(bool_ra_map_ca_cfar==True)[0].shape) # b4 OS-CFAR
        # print(np.where(bool_ra_map_os_cfar==True)[0].shape) # after OS-CFAR

        '''
        Step 3. Padding along R & A
        '''
        list_bool_ra_map = []
        if step == 2:
            list_bool_ra_map.append(bool_ra_map_os_cfar)
            ind_r_from_ra_os, ind_a_from_ra_os = np.where(bool_ra_map_os_cfar==True)
        elif step in [0,3]:
            pad_half_r, pad_half_a = cfg_os_cfar.PADDING_HALF_RA # 2, 1
            pad_r = 2*pad_half_r+1
            pad_a = 2*pad_half_a+1
            
            float_ra_map_os_cfar = bool_ra_map_os_cfar.astype(np.float64)
            float_mask_ra = np.ones((pad_r,pad_a), dtype=np.float64)
            float_ra_convolved = ndimage.convolve(float_ra_map_os_cfar, float_mask_ra, mode='constant')
            bool_ra_convolved = float_ra_convolved.astype(bool)
            ind_r_from_ra_os, ind_a_from_ra_os = np.where(bool_ra_convolved==True)

            if step == 3:
                list_bool_ra_map.append(bool_ra_convolved)
            elif step == 0: # all
                list_bool_ra_map.append(bool_ra_map_os_cfar)
                list_bool_ra_map.append(bool_ra_convolved)

        list_val_indicator = []
        for bool_ra_map_temp in list_bool_ra_map:
            ind_r_from_ra_os, ind_a_from_ra_os = np.where(bool_ra_map_temp==True)
            bool_rae_map_os_cfar_only_ra = np.full_like(rae_map, False, dtype=bool)
            bool_rae_map_os_cfar_only_ra[ind_r_from_ra_os,ind_a_from_ra_os,:] = True

            bool_rae_map_ca_cfar = np.full_like(rae_map, False, dtype=bool)
            bool_rae_map_ca_cfar[ind_r,ind_a,ind_e] = True
            bool_rae_map_os_cfar = np.logical_and(bool_rae_map_os_cfar_only_ra, bool_rae_map_ca_cfar)

            indicator_rae_map_os_cfar = np.full_like(rae_map, -1.)
            indicator_rae_map_os_cfar[ind_r,ind_a,ind_e] = 0.
            ind_r_os, ind_a_os, ind_e_os = np.where(bool_rae_map_os_cfar==True)
            indicator_rae_map_os_cfar[ind_r_os,ind_a_os,ind_e_os] = 1.
            val_indicator = indicator_rae_map_os_cfar[ind_r,ind_a,ind_e]
            list_val_indicator.append(val_indicator.reshape(-1,1).copy())

        '''
        if step in [2,3]:
            idx 5 is indicator, where 1.&0. are results of OS&CA-CFAR, respectively
            output is Nx6
        elif step == 0:
            idx 5 for step 2 & idx 6 for step 3
            output is Nx7
        ''' 
        list_concat = [pts_ca_cfar]
        list_concat.extend(list_val_indicator)
        
        return np.concatenate(list_concat, axis=1) # Nx(6or7)

    def _get_ca_cfar_idx_from_cube(self, cube_pw:np.array)->tuple:
        cfg_cfar = self.cfg_rdr.CA_CFAR
        cube_pw_norm = cube_pw/1.e+13 # preventing overflow

        nh_g_x, nh_g_y, nh_g_z = cfg_cfar.GUARD_CELL_RAE
        nh_t_x, nh_t_y, nh_t_z = cfg_cfar.TRAIN_CELL_RAE 
        mask_size = (2*(nh_g_x+nh_t_x)+1, 2*(nh_g_y+nh_t_y)+1, 2*(nh_g_z+nh_t_z)+1) # 1 for own
        mask = np.ones(mask_size)
        mask[nh_t_x:nh_t_x+2*nh_g_x+1, nh_t_y:nh_t_y+2*nh_g_y+1, nh_t_z:nh_t_z+2*nh_g_z+1] = 0
        num_total_train_cells = np.count_nonzero(mask)
        mask = mask/num_total_train_cells
        
        conv_out = ndimage.convolve(cube_pw_norm, mask, mode='constant')
        alpha = num_total_train_cells * (cfg_cfar.FALSE_ALARM_RATE**(-1/num_total_train_cells)-1)
        conv_out = alpha * conv_out
        bool_cfar_target = np.greater(cube_pw_norm, conv_out)
        
        return np.where(bool_cfar_target==True)
    
    def _get_ca_cfar_idx_from_ra_matrix(self, matrix_pw:np.array)->tuple:
        cfg_cfar_ra = self.cfg_rdr.CA_CFAR_RA
        matrix_pw_norm = matrix_pw/1.e+13 # preventing overflow

        nh_g_x, nh_g_y = cfg_cfar_ra.GUARD_CELL_RA
        nh_t_x, nh_t_y = cfg_cfar_ra.TRAIN_CELL_RA 
        mask_size = (2*(nh_g_x+nh_t_x)+1, 2*(nh_g_y+nh_t_y)+1) # 1 for own
        mask = np.ones(mask_size)
        mask[nh_t_x:nh_t_x+2*nh_g_x+1, nh_t_y:nh_t_y+2*nh_g_y+1] = 0
        num_total_train_cells = np.count_nonzero(mask)
        mask = mask/num_total_train_cells
        
        conv_out = ndimage.convolve(matrix_pw_norm, mask, mode='constant')
        alpha = num_total_train_cells * (cfg_cfar_ra.FALSE_ALARM_RATE**(-1/num_total_train_cells)-1)
        conv_out = alpha * conv_out
        bool_cfar_target = np.greater(matrix_pw_norm, conv_out)
        
        return np.where(bool_cfar_target==True)
    
    def update_calib_info(self, path_calib:str):
        self.list_calib_xyz = self._get_calib_info(path_calib)
    
    def show_rendered_rdr_bev_img(self, path_rdr_tesseract:str, path_label:str=None):
        arr_tesseract = self._load_rdr_tesseract(path_rdr_tesseract)
        img_bev = self._get_rendered_bev_img(arr_tesseract)

        if not (self.list_calib_xyz is None):
            self.object_renderer.update_calib_xyz(self.list_calib_xyz)
        if not (path_label is None):
            self.object_renderer.update_list_render_obj_from_path_label(path_label)
            img_bev = self.object_renderer.render_to_img_bev(img_bev,\
                self.cfg.RENDER_RDR_BEV.ROI_X, self.cfg.RENDER_RDR_BEV.ROI_Y)

        cv2.imshow('bev_img', img_bev)
        cv2.waitKey(0)

    def show_rendered_lpc(self, path_pcd:str, mode:str='pc', path_label:str=None, sampling_rate:float=None):
        pcd = self._load_ldr_pcd(path_pcd, sampling_per_azimuth=sampling_rate)
        if not (self.list_calib_xyz is None): # LiDAR is already calibrated
            self.object_renderer.update_calib_xyz(self.list_calib_xyz)
        if not (path_label is None):
            self.object_renderer.update_list_render_obj_from_path_label(path_label)

        if mode == 'pc':
            list_pcd = [pcd._get_o3d_pcd()]
            if not (path_label is None):
                list_pcd = self.object_renderer.render_to_list_pcd(list_pcd)
            o3d.visualization.draw_geometries(list_pcd)
        elif mode == 'bev':
            img_bev = pcd._get_bev_pcd(self.cfg.RENDER_LDR_BEV)
            if not (path_label is None):
                img_bev = self.object_renderer.render_to_img_bev(img_bev,\
                    self.cfg.RENDER_LDR_BEV.ROI_X, self.cfg.RENDER_LDR_BEV.ROI_Y)
            cv2.imshow('LiDAR pc in bev', img_bev)
            cv2.waitKey(0)

    def show_rendered_both_rpc_and_lpc(self, path_pcd:str, path_tesseract:str, path_label:str=None, sampling_rate:float=None):
        pcd = self._load_ldr_pcd(path_pcd, sampling_per_azimuth=sampling_rate)
        if not (self.list_calib_xyz is None): # LiDAR is already calibrated
            self.object_renderer.update_calib_xyz(self.list_calib_xyz)
        if not (path_label is None):
            self.object_renderer.update_list_render_obj_from_path_label(path_label)

        list_pcd = [pcd._get_o3d_pcd()]
        if not (path_label is None):
            list_pcd = self.object_renderer.render_to_list_pcd(list_pcd)
        o3d.visualization.draw_geometries(list_pcd)

        # 3D Radar
        rgb_3d_rpc = [0.,0.,0.]
        processed_3d_rpc = self._get_processed_3D_rdr_from_path_tesseract(path_tesseract)
        pcd_rpc = o3d.geometry.PointCloud()
        pcd_rpc.points = o3d.utility.Vector3dVector(processed_3d_rpc[:,:3])
        pcd_rpc.colors = o3d.utility.Vector3dVector(\
            np.array(rgb_3d_rpc).reshape(-1,3).repeat(len(processed_3d_rpc),0))
        list_pcd_3d = []
        list_pcd_3d.extend(list_pcd)
        list_pcd_3d.append(pcd_rpc)
        o3d.visualization.draw_geometries(list_pcd_3d)

        # 4D Radar
        processed_rpc = self._get_processed_rdr_from_path_tesseract(path_tesseract, step=0) # Nx7 (idx 5&6 for step 2&3 respectively)
        cfg_render_rpc = self.cfg.RENDER_RPC
        step = cfg_render_rpc.STEP
        if step == 1:
            rpc = processed_rpc[:,:5]
        elif step == 2:
            rpc = processed_rpc[np.where(processed_rpc[:,5]==1.)][:,:5]
        elif step == 3:
            rpc = processed_rpc[np.where(processed_rpc[:,6]==1.)][:,:5]
        else:
            rpc = processed_rpc[np.where(processed_rpc[:,6]==1.)][:,:5]

        pcd_rpc = o3d.geometry.PointCloud()
        pcd_rpc.points = o3d.utility.Vector3dVector(rpc[:,:3])
        pcd_rpc.colors = o3d.utility.Vector3dVector(\
            np.array([0.,0.,0.]).reshape(-1,3).repeat(len(rpc),0))
        list_pcd_4d = []
        list_pcd_4d.extend(list_pcd)
        list_pcd_4d.append(pcd_rpc)
        o3d.visualization.draw_geometries(list_pcd_4d)

    def show_rendered_processed_3d_rpc(self, path_tesseract:str, path_pcd:str=None, path_label:str=None):
        rgb_3d_rpc = [0.,0.,0.]

        processed_3d_rpc = self._get_processed_3D_rdr_from_path_tesseract(path_tesseract)
        if not (self.list_calib_xyz is None): # LiDAR is already calibrated
            self.object_renderer.update_calib_xyz(self.list_calib_xyz)
        if not (path_label is None):
            self.object_renderer.update_list_render_obj_from_path_label(path_label)
        pcd_rpc = o3d.geometry.PointCloud()
        pcd_rpc.points = o3d.utility.Vector3dVector(processed_3d_rpc[:,:3])
        pcd_rpc.colors = o3d.utility.Vector3dVector(\
            np.array(rgb_3d_rpc).reshape(-1,3).repeat(len(processed_3d_rpc),0))
        list_pcd = []
        if not (path_pcd is None):
            pcd_lpc = self._load_ldr_pcd(path_pcd)
            list_pcd.append(pcd_lpc._get_o3d_pcd())
        list_pcd.append(pcd_rpc)
        list_pcd = self.object_renderer.render_to_list_pcd(list_pcd)
        o3d.visualization.draw_geometries(list_pcd)

    def show_rendered_processed_rpc(self, path_tesseract:str, mode:str='pc', path_pcd:str=None, path_label:str=None):
        processed_rpc = self._get_processed_rdr_from_path_tesseract(path_tesseract, step=0) # Nx7 (idx 5&6 for step 2&3 respectively)
        
        if not (self.list_calib_xyz is None): # LiDAR is already calibrated
            self.object_renderer.update_calib_xyz(self.list_calib_xyz)
        if not (path_label is None):
            self.object_renderer.update_list_render_obj_from_path_label(path_label)

        cfg_render_rpc = self.cfg.RENDER_RPC
        step = cfg_render_rpc.STEP
        dict_rgb_step = cfg_render_rpc.RGB_STEP
        if step == 1:
            rpc = processed_rpc[:,:5]
        elif step == 2:
            rpc = processed_rpc[np.where(processed_rpc[:,5]==1.)][:,:5]
        elif step == 3:
            rpc = processed_rpc[np.where(processed_rpc[:,6]==1.)][:,:5]
        else:
            rpc = processed_rpc[np.where(processed_rpc[:,6]==1.)][:,:5]

        if mode == 'pc':
            pcd_rpc = o3d.geometry.PointCloud()
            pcd_rpc.points = o3d.utility.Vector3dVector(rpc[:,:3])
            pcd_rpc.colors = o3d.utility.Vector3dVector(\
                np.array(dict_rgb_step[f'STEP{step}']).reshape(-1,3).repeat(len(rpc),0))
            list_pcd = []
            if not (path_pcd is None):
                pcd_lpc = self._load_ldr_pcd(path_pcd)
                list_pcd.append(pcd_lpc._get_o3d_pcd())
            list_pcd.append(pcd_rpc)
            list_pcd = self.object_renderer.render_to_list_pcd(list_pcd)
            o3d.visualization.draw_geometries(list_pcd)
        elif mode == 'bev':
            if path_pcd is None: # img_bev is radar
                img_bev = self._get_rendered_bev_img(self._load_rdr_tesseract(path_tesseract))
                x_min, _, x_max = self.cfg_render_rdr_bev.ROI_X
                y_min, _, y_max = self.cfg_render_rdr_bev.ROI_Y
            else:
                pcd_lpc = self._load_ldr_pcd(path_pcd)
                img_bev = pcd_lpc._get_bev_pcd(self.cfg.RENDER_LDR_BEV)
                x_min, _, x_max = self.cfg_render_rdr_bev.ROI_X
                y_min, _, y_max = self.cfg_render_rdr_bev.ROI_Y
            
            x_size, y_size = cfg_render_rpc.IMG_SIZE_BEV_XY
            img_bev = cv2.resize(img_bev, (y_size, x_size))
            x_bin = (x_max-x_min)/x_size
            y_bin = (y_max-y_min)/y_size
            arr_x = np.linspace(x_min, x_max-x_bin, x_size) + x_bin/2.
            arr_y = np.linspace(y_min, y_max-y_bin, y_size) + y_bin/2.

            def get_rpc_bev(temp_rpc:np.array, temp_rgb:list)->np.array:
                x_pix = ((temp_rpc[:,0]-x_min + x_bin/2.)/x_bin).astype(int)
                y_pix = ((temp_rpc[:,1]-y_min + y_bin/2.)/y_bin).astype(int)

                val_pix = np.log10(temp_rpc[:,3]) # value regarding pw
                min_val_pix = np.min(val_pix)
                val_pix = 128 + ((val_pix-min_val_pix)\
                    /(np.max(val_pix)-min_val_pix)*127.).astype(np.uint8).astype(int)

                temp_hsv = cv2.cvtColor((np.array(temp_rgb)*255.).reshape(1,1,3)\
                                        .astype(np.uint8), code=cv2.COLOR_RGB2HSV)
                
                # temp_xyval = np.column_stack((x_pix,y_pix,val_pix))
                # temp_xyval = temp_xyval[np.where(
                #     (temp_xyval[:,0]>0) & (temp_xyval[:,0]<len(arr_x)) &
                #     (temp_xyval[:,1]>0) & (temp_xyval[:,1]<len(arr_y))
                # )]

                temp_black_img = np.zeros((len(arr_x), len(arr_y)), dtype=np.uint8)
                temp_black_img[x_pix,y_pix] = val_pix.astype(np.uint8)

                dilation = cfg_render_rpc.DILATION
                if dilation != 0:
                    temp_black_img = cv2.dilate(temp_black_img, (dilation, dilation))

                temp_hsv = temp_hsv.repeat(len(arr_x),0).repeat(len(arr_y),1)
                temp_hsv[:,:,2] = temp_black_img
                temp_hsv = cv2.cvtColor(temp_hsv, cv2.COLOR_HSV2BGR)
                temp_hsv = np.flip(temp_hsv, (0,1))

                return temp_hsv

            if step in [1,2,3]:
                img_rpc = get_rpc_bev(rpc, dict_rgb_step[f'STEP{step}'])
                ind_valid_u, ind_valid_v = np.where(np.sum(img_rpc,2)!=0)
                img_bev = self.object_renderer.render_to_img_bev(img_bev, [x_min,x_bin,x_max], [y_min,y_bin,y_max])
                img_bev[ind_valid_u,ind_valid_v,:] = img_rpc[ind_valid_u,ind_valid_v,:]
                cv2.imshow('bev', img_bev)
                cv2.waitKey(0)

            elif step == 0:
                rpc1 = processed_rpc[:,:5]
                rpc2 = processed_rpc[np.where(processed_rpc[:,5]==1.)][:,:5]
                rpc3 = processed_rpc[np.where(processed_rpc[:,6]==1.)][:,:5]

                img_bev_0 = self.object_renderer.render_to_img_bev(img_bev, [x_min,x_bin,x_max], [y_min,y_bin,y_max])
                img_rpc1 = get_rpc_bev(rpc1, dict_rgb_step[f'STEP1'])
                img_rpc2 = get_rpc_bev(rpc2, dict_rgb_step[f'STEP2'])
                img_rpc3 = get_rpc_bev(rpc3, dict_rgb_step[f'STEP3'])

                img_bev_1 = img_bev_0.copy()
                ind_valid_u1, ind_valid_v1 = np.where(np.sum(img_rpc1,2)!=0)
                img_bev_1[ind_valid_u1,ind_valid_v1,:] = img_rpc1[ind_valid_u1,ind_valid_v1,:]

                img_bev_2 = img_bev_0.copy()
                ind_valid_u2, ind_valid_v2 = np.where(np.sum(img_rpc2,2)!=0)
                img_bev_2[ind_valid_u2,ind_valid_v2,:] = img_rpc2[ind_valid_u2,ind_valid_v2,:]

                img_bev_3 = img_bev_0.copy()
                ind_valid_u3, ind_valid_v3 = np.where(np.sum(img_rpc3,2)!=0)
                img_bev_3[ind_valid_u3,ind_valid_v3,:] = img_rpc3[ind_valid_u3,ind_valid_v3,:]

                cv2.imshow('bev', cv2.hconcat([img_bev_0,img_bev_1,img_bev_2,img_bev_3]))
                cv2.waitKey(0)
    
    ### For making movie with loaded imgs ###
    def setting_for_processed_imgs(self, dict_directory:dict):
        cfg_dir = dict_directory
        list_dict_item = []
        if cfg_dir.DIR.SEQ_LOAD is None:
            list_seqs = os.listdir(cfg_dir.DIR.DIR_INF)
        else:
            list_seqs = cfg_dir.DIR.SEQ_LOAD
        self.list_seqs = list_seqs
        dir_inf = cfg_dir.DIR.DIR_INF
        self.dir_rdr = cfg_dir.DIR.DIR_RENDERED_RDR
        self.dir_ldr = cfg_dir.DIR.DIR_RENDERED_LDR
        self.dir_camf = cfg_dir.DIR.DIR_PROCESSED_CAM
        self.dir_srt = cfg_dir.DIR.DIR_FILT_SRT
        for seq_name in list_seqs:
            list_frames = sorted(os.listdir(osp.join(dir_inf, seq_name)))
            for file_name in list_frames:
                path_inf = osp.join(dir_inf, seq_name, file_name)
                list_dict_item.append(self._load_inf_info(path_inf))
        self.cfg_dir = cfg_dir
        self.list_dict_item = list_dict_item
        self.proj = np.load(osp.join(self.dir_camf, 'proj.npy'))
        self.dir_save = cfg_dir.DIR_SAVE

    def _load_inf_info(self, path_inf)->dict:
        f = open(path_inf)
        lines = f.readlines()
        f.close()
        lines = list(map(lambda x: x.rstrip('\n'), lines))
        header = lines[0]
        objs = lines[2:]
        list_gt_render_obj = []
        list_pred_render_obj = []
        is_gt = True
        for temp_obj in objs:
            if temp_obj == '-':
                is_gt = False
            else:
                list_vals = temp_obj.split(',')
                idx = int(list_vals[0])
                list_obj_info = list(map(lambda x: float(x), list_vals[2:9]))
                if is_gt:
                    cls_name = 'Label'
                    type_str = 'gt'
                    render_obj = RenderObject3D(type_str, idx, cls_name, list_obj_info)
                    list_gt_render_obj.append(render_obj)
                else:
                    cls_name = list_vals[1]
                    type_str = 'pred'
                    render_obj = RenderObject3D(type_str, idx, cls_name, list_obj_info, float(list_vals[9]))
                    list_pred_render_obj.append(render_obj)
        idx_str, tstamp_str = header.split(',')
        idx_rdr, idx_ldr, idx_camf, _, _ = idx_str.split('=')[1].split('_')
        tstamp = float(tstamp_str.split('=')[1])
        seq_name = path_inf.split('/')[-2]

        dict_path = {
            'inf': path_inf,
            'camf': osp.join(self.dir_camf, seq_name, f'front_{idx_camf}.npy'),
            'rdr_bev': osp.join(self.dir_rdr, seq_name, f'tesseract_{idx_rdr}.png'),
            'ldr_bev': osp.join(self.dir_ldr, seq_name, f'os2-64_{idx_ldr}.png'),
            'rdr_srt': osp.join(self.dir_srt, 'temp', seq_name, f'rp_{idx_rdr}.npy')
        }
        dict_item = {
            'path': dict_path,
            'tstamp': tstamp,
            'pred': list_pred_render_obj,
            'gt': list_gt_render_obj,
        }
        return dict_item
    
    def _get_rendered_img_from_preprocessed_img(self, dict_item:dict,
            type_sensor:str='rdr', is_w_label:bool=True, is_w_pred:bool=True)->np.array:
        '''
        type_sensor
            in ['rdr', 'ldr', 'camf']
        '''
        self.object_renderer.clear_list_render_obj()
        if is_w_label:
            self.object_renderer.extend_list_render_obj_from_obj(dict_item['gt'])
        if is_w_pred:
            self.object_renderer.extend_list_render_obj_from_obj(dict_item['pred'])
        
        if type_sensor in ['rdr', 'ldr']:
            img_bev = cv2.imread(dict_item['path'][type_sensor+'_bev'])
            roi_x = self.cfg.RENDER_LOAD.ROI_X
            roi_y = self.cfg.RENDER_LOAD.ROI_Y
            if type_sensor == 'rdr':
                img_bev = np.flip(img_bev, axis=1)
            img_bev = self.object_renderer.render_to_img_bev(img_bev, roi_x, roi_y)
            img = img_bev
        elif type_sensor == 'camf':
            img_camf = np.load(dict_item['path']['camf'])
            img_camf = self.object_renderer.render_to_img_perspective(img_camf, self.proj)
            img = img_camf
            
        return img
    
    def show_rendered_img_from_preprocessed_img(self, dict_item:dict,
            type_sensor:str='all', is_w_label:bool=True, is_w_pred:bool=True, msec:int=0, is_save:bool=False):
        '''
        type_sensor
            in ['rdr', 'ldr', 'camf']
        '''
        if type_sensor == 'all':
            img_camf = self._get_rendered_img_from_preprocessed_img(dict_item, 'camf', is_w_label, is_w_pred)
            img_ldr = self._get_rendered_img_from_preprocessed_img(dict_item, 'ldr', is_w_label, is_w_pred)
            img_rdr = self._get_rendered_img_from_preprocessed_img(dict_item, 'rdr', is_w_label, is_w_pred)
            
            list_imgs = [img_camf, img_ldr, img_rdr]
            min_height = min(img_camf.shape[0], img_ldr.shape[0], img_rdr.shape[0])
            for idx, temp_img in enumerate(list_imgs):
                temp_h, temp_w, _ = temp_img.shape
                list_imgs[idx] = cv2.resize(temp_img, (int(temp_w*min_height/float(temp_h)+0.5), min_height))

            img = cv2.hconcat(list_imgs)
        else:
            img = self._get_rendered_img_from_preprocessed_img(dict_item, type_sensor, is_w_label, is_w_pred)

        if is_save:
            path_inf = dict_item['path']['inf']
            list_split = path_inf.split('/')
            setting_name = list_split[-5]
            model_name = list_split[-4]
            conf_name = list_split[-3][4:]
            seq_name = list_split[-2]
            img_name = 'img_'+((list_split[-1]).split('.')[0])+'.png'
            temp_dir = osp.join(self.dir_save, setting_name, seq_name, model_name+conf_name)
            os.makedirs(temp_dir, exist_ok=True)
            cv2.imwrite(osp.join(temp_dir, img_name), img)
        else:
            cv2.imshow('img', img)
            cv2.waitKey(msec)
    ### For making movie with loaded imgs ###
    
    ### For showing data with 4D Radar tensors ###
    def setting_for_rdr_tesseract(self, dict_dir_rdr_tesseract:dict):
        cfg_dir = dict_dir_rdr_tesseract
        list_dir = cfg_dir.DIR.LIST_DIR
        seq_load = cfg_dir.DIR.SEQ_LOAD
        dir_revised_label = cfg_dir.DIR.DIR_REVISED_LABEL
        dir_camf = cfg_dir.DIR.DIR_PROCESSED_CAM
        dir_rendered_rdr = cfg_dir.DIR.DIR_RENDERED_RDR
        is_use_revised_label = False if dir_revised_label is None else True
        list_dict_item = []
        for temp_dir in list_dir:
            list_seq = os.listdir(temp_dir)
            for seq in list_seq:
                if seq_load is None:
                    pass
                elif not (seq in seq_load):
                    continue
                list_temp_label = sorted(os.listdir(osp.join(temp_dir, seq, 'info_label')))
                for temp_label in list_temp_label:
                    path_label = osp.join(dir_revised_label, seq+'_info_label_revised', temp_label)\
                        if is_use_revised_label else osp.join(temp_dir, seq, temp_label)
                    f = open(path_label, 'r')
                    line = f.readline()
                    f.close()
                    idx_str_rdr, idx_str_ldr, idx_str_camf, _, _ = line.split(',')[0].split('=')[1].split('_')
                    dict_path = {
                        'tesseract': osp.join(temp_dir, seq, 'radar_tesseract', f'tesseract_{idx_str_rdr}.mat'),
                        'lpc': osp.join(temp_dir, seq, 'os2-64', f'os2-64_{idx_str_ldr}.pcd'),
                        'camf': osp.join(dir_camf, seq, f'front_{idx_str_camf}.npy'),
                        'calib': osp.join(temp_dir, seq, 'info_calib', 'calib_radar_lidar.txt'),
                        'proj': osp.join(dir_camf, 'proj.npy'),
                        'label': path_label,
                        'rdr_bev': osp.join(dir_rendered_rdr, seq, f'tesseract_{idx_str_rdr}.png')
                    }
                    dict_item = {
                        'path': dict_path,
                    }
                    list_dict_item.append(dict_item)
        self.list_tesseract_item = list_dict_item

    def show_rendered_data_from_tesseract(self, save_dir:str=None):
        for idx_item, dict_item in enumerate(tqdm(self.list_tesseract_item)):
            # if idx_item < 450:
            #     continue
            dict_path = dict_item['path']

            path_calib = dict_path['calib']
            self.update_calib_info(path_calib)

            path_tesseract = dict_path['tesseract']
            path_pcd = dict_path['lpc']
            path_camf = dict_path['camf']
            path_label = dict_path['label']

            self.object_renderer.clear_list_render_obj()
            self.object_renderer.update_calib_xyz(self.list_calib_xyz)
            self.object_renderer.update_list_render_obj_from_path_label(path_label)

            '''
            (1) 3D Radar vs 4D Radar
            '''
            self.show_rendered_processed_3d_rpc(path_tesseract, path_pcd, path_label)
            self.show_rendered_processed_rpc(path_tesseract, 'pc', path_pcd, path_label)
            '''
            (2) Front cam img
            '''
            img_camf = np.load(path_camf)
            img_camf = self.object_renderer.render_to_img_perspective(img_camf,\
                                                np.load(dict_item['path']['proj']))
            if save_dir is None:
                cv2.imshow('img_camf', img_camf)
                cv2.waitKey(0)
            else:
                path_save = osp.join(save_dir, path_camf.split('/')[-1].split('.')[0]+'.png')
                cv2.imwrite(path_save, img_camf)
            '''
            (3) Radar BEV
            '''
            # img_bev = cv2.imread(dict_item['path']['rdr_bev'])
            # roi_x = self.cfg.RENDER_LOAD.ROI_X
            # roi_y = self.cfg.RENDER_LOAD.ROI_Y
            # img_bev = np.flip(img_bev, axis=1)
            # img_bev = self.object_renderer.render_to_img_bev(img_bev, roi_x, roi_y)
            # if save_dir is None:
            #     cv2.imshow('rdr_bev', img_bev)
            #     cv2.waitKey(0)
            # else:
            #     path_save = osp.join(save_dir, dict_item['path']['rdr_bev'].split('/')[-1])
            #     cv2.imwrite(path_save, img_bev)

    ### For showing data with 4D Radar tensors ###
    
if __name__ == '__main__':
    # DICT_TEMP_PATH = {
    #     'tesseract':    '/media/donghee/HDD_3/tesseract_21/tesseract_00005.mat',
    #     'lpc':          '/media/donghee/HDD_3/tesseract_21/os2-64_00001.pcd',
    #     'camf':         '/media/donghee/HDD_3/tesseract_21/front_00004.npy',
    #     'calib':        '/media/donghee/HDD_3/tesseract_21/calib_radar_lidar.txt',
    #     'proj':         '/media/donghee/HDD_3/tesseract_21/proj.npy',
    #     'label':        '/media/donghee/HDD_3/tesseract_21/00005_00001.txt',
    # }
    DICT_TEMP_PATH = {
        'tesseract':    '/media/donghee/HDD_3/tesseract_seq10/tesseract_00065.mat',
        'lpc':          '/media/donghee/HDD_3/tesseract_seq10/os2-64_00039.pcd',
        'camf':         '/media/donghee/HDD_3/tesseract_seq10/front_00004.npy',
        'calib':        '/media/donghee/HDD_3/tesseract_seq10/calib_radar_lidar.txt',
        'proj':         '/media/donghee/HDD_3/tesseract_seq10/proj.npy',
        'label':        '/media/donghee/HDD_3/tesseract_seq10/00065_00039.txt',
    }
    DICT_DIRECTORY = {
        'DIR': {
            'LIST_DIR': ['/media/donghee/HDD_3/K-Radar/radar_bin_lidar_bag_files/generated_files'],
            'DIR_FILT_SRT': '/media/donghee/HDD_3/K-Radar/dir_sp_filtered',
            'DIR_REVISED_LABEL': '/media/donghee/HDD_3/K-Radar/kradar_revised_label',
            'DIR_PROCESSED_CAM': '/media/donghee/HDD_3/K-Radar/dir_processed_img/img_driving_corridor',
            'DIR_RENDERED_LDR': '/media/donghee/HDD_3/K-Radar/dir_rendered_lpc',
            'DIR_RENDERED_RDR': '/media/donghee/HDD_3/KRadar_radar_frames',
            'DIR_INF': '/media/donghee/HDD_3/K-Radar_inf_results/driving_corridor_only_sedan/RC_FUSION/conf_0_3',
            'SEQ_LOAD': ['12'], # ['50'], # None: Loading all seqs
        },
        'DIR_SAVE': '/media/donghee/HDD_3/K-Radar_rendered'
    }
    DICT_DIR_RDR_TESSERACT = { # Lab server
        'DIR': {
            'LIST_DIR': ['/media/ave/HDD_3_1/gen_2to5',
                         '/media/ave/HDD_3_1/radar_bin_lidar_bag_files/generated_files',
                         '/media/ave/HDD_3_2/radar_bin_lidar_bag_files/generated_files',
                         '/media/ave/data_2/radar_bin_lidar_bag_files/generated_files'],
            'DIR_REVISED_LABEL': '/media/ave/HDD_3_1/kradar_revised_label',
            'DIR_RENDERED_RDR': None,
            'DIR_PROCESSED_CAM': '/media/ave/HDD_3_1/dir_processed_img/img_driving_corridor',
            'SEQ_LOAD': ['12'],
        },
        'DIR_SAVE': '/media/ave/HDD_3_2/K-Radar_render_tesseract'
    }

    radar_preprocessor = RadarPreprocessor(cfg=EasyDict(CFG_DICT))
    radar_preprocessor.update_calib_info(DICT_TEMP_PATH['calib']) # for label & lpc

    ### Temp ###
    # img_camf = np.load(DICT_TEMP_PATH['camf'])
    # radar_preprocessor.object_renderer.update_calib_xyz(radar_preprocessor.list_calib_xyz)
    # radar_preprocessor.object_renderer.update_list_render_obj_from_path_label(DICT_TEMP_PATH['label'])
    # img_camf = radar_preprocessor.object_renderer.render_to_img_perspective(img_camf, np.load(DICT_TEMP_PATH['proj']))
    # cv2.imshow('camf', img_camf)
    # cv2.waitKey(0)
    ### Temp ###
    
    ### Raw data -> Render: uncomment update_calib_info ###
    # radar_preprocessor.show_rendered_rdr_bev_img(DICT_TEMP_PATH['tesseract'], DICT_TEMP_PATH['label'])
    radar_preprocessor.show_rendered_both_rpc_and_lpc(DICT_TEMP_PATH['lpc'], DICT_TEMP_PATH['tesseract'], DICT_TEMP_PATH['label'], sampling_rate=0.4)
    # radar_preprocessor.show_rendered_lpc(DICT_TEMP_PATH['lpc'], 'pc', DICT_TEMP_PATH['label'], sampling_rate=0.05) # ['bev', 'pc']

    # radar_preprocessor.show_rendered_processed_rpc(DICT_TEMP_PATH['tesseract'], 'bev', None, DICT_TEMP_PATH['label'])
    # radar_preprocessor.show_rendered_processed_rpc(DICT_TEMP_PATH['tesseract'], 'bev', DICT_TEMP_PATH['lpc'], DICT_TEMP_PATH['label'])
    # radar_preprocessor.show_rendered_processed_rpc(DICT_TEMP_PATH['tesseract'], 'pc', DICT_TEMP_PATH['lpc'], DICT_TEMP_PATH['label'])
    # radar_preprocessor.show_rendered_processed_3d_rpc(DICT_TEMP_PATH['tesseract'], DICT_TEMP_PATH['lpc'], DICT_TEMP_PATH['label'])
    ### Raw data -> Render: uncomment update_calib_info ###
    
    ### Image -> Render: comment update_calib_info ###
    # radar_preprocessor.setting_for_processed_imgs(EasyDict(DICT_DIRECTORY))
    # for idx, dict_item in enumerate(tqdm(radar_preprocessor.list_dict_item)):
    #     # if idx < 11640:
    #     #     continue
    #     try:
    #         radar_preprocessor.show_rendered_img_from_preprocessed_img(dict_item, 'rdr', msec=0, is_w_label=False, is_w_pred=True)
    #         # radar_preprocessor.show_rendered_img_from_preprocessed_img(dict_item, 'all', is_save=True)
    #     except:
    #         path_inf = dict_item['path']['inf']
    #         print(f'error in {path_inf}')
    #         continue
    ### Image -> Render: comment update_calib_info ###

    ### Tesseract -> Render: uncomment update_calib_info ###
    # radar_preprocessor.setting_for_rdr_tesseract(EasyDict(DICT_DIRECTORY))
    # radar_preprocessor.show_rendered_data_from_tesseract()#save_dir='/media/donghee/HDD_3/K-Radar_render_tesseract/front_img')
    # radar_preprocessor.show_rendered_lpc()
    ### Tesseract -> Render: uncomment update_calib_info ###
