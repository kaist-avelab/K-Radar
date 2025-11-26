'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
* comment: calibration for camera & augmentation
'''

import os
import os.path as osp
import numpy as np
import yaml
import open3d as o3d
import torch
import cv2
import torchvision
import copy
import pickle

from tqdm import tqdm
from easydict import EasyDict
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import Dataset

try:
    from utils.util_calib import *
    from utils.util_dataset import \
        func_save_undistorted_camera_imgs_w_projected_params, \
        func_save_depth_labels_for_cams, \
        func_get_distribution_of_label, \
        func_save_occupied_bev_map
except:
    import sys
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util_calib import *
    from utils.util_dataset import \
        func_save_undistorted_camera_imgs_w_projected_params, \
        func_save_depth_labels_for_cams, \
        func_get_distribution_of_label, \
        func_save_occupied_bev_map

roi = [0,-16,-2,72,16,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/donghee/kradar/dataset'],
        split = ['./resources/split/train.txt', './resources/split/test.txt'],
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
    ),
    label = { # (consider, logit_idx, rgb, bgr)
        'calib':            True,
        'onlyR':            False,
        'consider_cls':     False,
        'consider_roi':     False,
        'remove_0_obj':     False,
        'Sedan':            [True,  1,  [0, 1, 0],       [0,255,0]],
        'Bus or Truck':     [True,  2,  [1, 0.2, 0],     [0,50,255]],
        'Motorcycle':       [False, -1, [1, 0, 0],       [0,0,255]],
        'Bicycle':          [False, -1, [1, 1, 0],       [0,255,255]],
        'Bicycle Group':    [False, -1, [0, 0.5, 1],     [0,128,255]],
        'Pedestrian':       [False, -1, [0, 0, 1],       [255,0,0]],
        'Pedestrian Group': [False, -1, [0.4, 0, 1],     [255,0,100]],
        'Label':            [False, -1, [0.5, 0.5, 0.5], [128,128,128]],
    },
    label_version = 'v2_0',
    item = dict(calib=True, ldr64=True, ldr128=False, rdr=False, rdr_sparse=False, cam=True, rdr_polar_3d=False, rpcs=False),
    calib = dict(z_offset=0.7),
    cam = dict(front0=True, front1=False, left0=False, left1=False, right0=False, right1=False, rear0=False, rear1=False),
    cam_dir = dict(load='cropped', undistorted='/media/donghee/HDD_0/kradar_imgs/undistorted', \
                   cropped='/media/donghee/HDD_0/kradar_imgs/cropped'), # load in ['ori', 'undistorted', 'cropped']
    cam_process = dict(ori_shape=(1280,720), resize=0.7, crop=(96,170,800,426), mean=[0.303, 0.303, 0.307], std=[0.113, 0.119, 0.107]),
    cam_calib = dict(load=True, dir='./resources/cam_calib/calib_seq'),
    t_params = dict(load=True, dir='./resources/cam_calib/T_params_seq', ref_sensor='radar'),
    ldr64 = dict(processed=False, dir='media/donghee/HDD_0/processed_lpc_calib_roi', skip_line=13, n_attr=9, inside_ldr64=True, calib=True,),
    rdr = dict(cube=False,),
    rdr_sparse = dict(processed=True, dir='/media/donghee/kradar/rdr_sparse_data/rtnh_wider_1p_1',),
    rdr_polar_3d = dict(processed=False, dir='/media/donghee/kradar/rdr_polar_3d', in_pc100p=True),
    roi = dict(filter=False, xyz=roi, keys=['ldr64', 'rdr_sparse'], check_azimuth_for_rdr=True, azimuth_deg=[-53,53]),
    rpcs = dict(processed=False, dir='/media/donghee/kradar/rdr_pc', keys=['pc1p', 'pc10p']),
    depth_labels = dict(load=True, dir='/media/donghee/HDD_0/kradar_depth_labels/cropped', stride=1, img_size=(704,256), margin=1, d_minmax=[2.0, 80.0]),
    obj_mask = dict(load=True, dir='/media/donghee/HDD_0/kradar_obj_mask_1_5_expand'),
    img_arrays = dict(load=True, img_num=3, interval=1, dir='/media/donghee/HDD_2/kradar_imgs_all/cropped'),
    portion = None,
)

class KRadarFusion_v1_0_CalibChange(Dataset):
    def __init__(self, cfg=None, split='all', **kwargs):
        if cfg == None:
            cfg = EasyDict(dict_cfg)
            cfg_from_yaml = False
            self.cfg=cfg
        else:
            cfg_from_yaml = True
            self.cfg=cfg.DATASET

        self.label = self.cfg.label
        self.label_version = self.cfg.get('label_version', 'v2_0')
        self.load_label_in_advance = True if self.label.remove_0_obj else False
        
        self.item = self.cfg.item
        self.calib = self.cfg.calib
        self.cam = self.cfg.get('cam', None)
        self.cam_dir = self.cfg.get('cam_dir', None)
        self.cam_process = self.cfg.get('cam_process', None)
        self.cam_calib = self.cfg.get('cam_calib', None)
        self.t_params = self.cfg.get('t_params', None)
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.rdr_polar_3d = self.cfg.get('rdr_polar_3d', None)
        self.rpcs = self.cfg.get('rpcs', None)
        self.roi = self.cfg.roi
        self.depth_labels = self.cfg.get('depth_labels', None)
        
        if self.depth_labels is None:
            self.load_depth_labels = False
        else:
            self.load_depth_labels = self.depth_labels.load
            self.dir_depth_labels = self.depth_labels.dir
            self.stride_depth_labels = self.depth_labels.stride
            self.img_size_depth_labels = self.depth_labels.img_size
            self.margin_depth_labels = self.depth_labels.margin
            self.d_minmax_depth_labels = self.depth_labels.d_minmax

        self.obj_mask = self.cfg.get('obj_mask', None)

        if self.obj_mask is None:
            self.load_obj_mask = False
        else:
            self.load_obj_mask = self.obj_mask.load
            self.dir_obj_mask = self.obj_mask.dir

        # self.img_arrays = self.cfg.get('img_arrays', None)

        # if self.img_arrays is None:
        #     self.load_img_arrays = False
        # else:
        #     self.load_img_arrays = self.img_arrays.load
        #     self.dir_img_arrays = self.img_arrays.dir
        #     self.num_img_arrays = self.img_arrays.img_num
        #     self.interval_img_arrays = self.img_arrays.interval

        for temp_key in ['cam', 'rdr_polar_3d', 'rpcs']:
            if temp_key not in self.item.keys():
                self.item[temp_key] = False
        
        self.portion = self.cfg.get('portion', None)

        self.list_dict_item = self.load_dict_item(self.cfg.path_data, split)
        if cfg_from_yaml:
            self.cfg.NUM = len(self)
        
        self.collate_ver = self.cfg.get('collate_fn', 'v2_1') # Post-processing

        self.arr_range, self.arr_azimuth, self.arr_elevation, \
            self.arr_doppler = self.load_physical_values(is_with_doppler=True)
        
        arr_r = self.arr_range
        arr_a = self.arr_azimuth
        arr_e = self.arr_elevation

        r_min = np.min(arr_r)
        r_bin = np.mean(arr_r[1:]-arr_r[:-1])
        r_max = np.max(arr_r)
        
        a_min = np.min(arr_a)
        a_bin = np.mean(arr_a[1:]-arr_a[:-1])
        a_max = np.max(arr_a)

        e_min = np.min(arr_e)
        e_bin = np.mean(arr_e[1:]-arr_e[:-1])
        e_max = np.max(arr_e)

        self.info_rae = [
            [r_min, r_bin, r_max],
            [a_min, a_bin, a_max],
            [e_min, e_bin, e_max]]
        
        if self.cam_dir is None:
            self.img_load = 2 # original
        else:
            if self.cam_dir.load == 'cropped':
                self.img_load = 0
                self.img_dir = self.cam_dir.cropped
            elif self.cam_dir.load == 'undistorted':
                self.img_load = 1
                self.img_dir = self.cam_dir.undistorted
            else:
                self.img_load = 2

            self.list_dealt_cams = []
            for k, v in self.cam.items():
                if v:
                    self.list_dealt_cams.append(k)

        if self.cam_process is not None:
            # W, H = self.cam_process.ori_shape
            # fH, fW = self.cam_process.FINAL_DIM # 704, 256
            resize = self.cam_process.resize
            # resize_dims = (int(W * resize), int(H * resize))
            # newW, newH = resize_dims
            crop = self.cam_process.crop
            self.img_process_infos = [resize, crop, False, 0] # Flip, Rotate

            is_use_imgnet_mean_std = self.cam_process.get('is_use_imgnet_mean_std', False)
            rgb_mean = [0.485, 0.456, 0.406] if is_use_imgnet_mean_std else self.cam_process.mean
            rgb_std = [0.229, 0.224, 0.225] if is_use_imgnet_mean_std else self.cam_process.std
            self.compose_img = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std),
                ]
            )

        if self.cam_calib is not None:
            self.dict_cam_calib = self.get_dict_cam_calib_from_yml_all_seqs() \
                                if self.cam_calib.load else None
        
        if self.t_params is not None:
            self.dict_t_params = self.get_t_params_all_seqs() \
                                if self.t_params.load else None
        
        shuffle_points = self.cfg.get('shuffle_points', None)
        self.shuffle_points = False if shuffle_points is None else \
                                        shuffle_points.get(split, False)
        
        if self.rdr_polar_3d is not None:
            if self.rdr_polar_3d.get('in_pc100p', False):
                n_r = len(self.arr_range)
                n_a = len(self.arr_azimuth)
                n_e = len(self.arr_elevation)
                rae_r = np.repeat(np.repeat((self.arr_range).copy().reshape(n_r,1,1),n_a,1),n_e,2)
                rae_a = np.repeat(np.repeat((self.arr_azimuth).copy().reshape(1,n_a,1),n_r,0),n_e,2)
                rae_e = np.repeat(np.repeat((self.arr_elevation).copy().reshape(1,1,n_e),n_r,0),n_a,1)

                # Radar polar to General polar coordinate
                rae_a = -rae_a
                rae_e = -rae_e
                
                # For flipped azimuth & elevation angle
                xyz_x = rae_r * np.cos(rae_e) * np.cos(rae_a)
                xyz_y = rae_r * np.cos(rae_e) * np.sin(rae_a)
                xyz_z = rae_r * np.sin(rae_e)

                self.rdr_polar_3d_xyz = np.stack((xyz_x,xyz_y,xyz_z),axis=0)
                self.get_rdr_polar_3d_in_pc100p = True
            else:
                self.get_rdr_polar_3d_in_pc100p = False

        self.cfg_effect = kwargs.get('cfg_effect', None)
        if self.cfg_effect is not None:
            self.cfg_effect = EasyDict(self.cfg_effect)

        self.cfg_calib_change = kwargs.get('cfg_calib_change', None)
        if self.cfg_calib_change is not None:
            self.cfg_calib_change = EasyDict(self.cfg_calib_change)
    
    ### Setup ###
    def load_dict_item(self, path_data, split):
        def get_split(split_txt, list_dict_split, val):
            f = open(split_txt, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                seq, label = line.split(',')
                list_dict_split[int(seq)][label.rstrip('\n')] = val
        list_dict_split = [dict() for _ in range(58+1)]
        get_split(path_data.split[0], list_dict_split, 'train')
        get_split(path_data.split[1], list_dict_split, 'test')
        
        list_seqs_w_header = []
        for path_header in path_data.list_dir_kradar:
            list_seqs = os.listdir(path_header)
            if self.portion is None:
                list_seqs_w_header.extend([(seq, path_header) for seq in list_seqs])
            else:
                for seq in list_seqs:
                    if seq in self.portion:
                        list_seqs_w_header.append((seq, path_header))
        list_seqs_w_header = sorted(list_seqs_w_header, key=lambda x: int(x[0]))

        list_dict_item = []
        for seq, path_header in list_seqs_w_header:
            list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
            for label in list_labels:
                path_label_v1_0 = osp.join(path_header, seq, 'info_label', label)
                path_label_v1_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v1_1/{seq}_info_label_revised', label)
                path_label_v2_0 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_0', 'KRadar_refined_label_by_UWIPL', seq, label)
                path_label_v2_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_1', 'KRadar_revised_visibility', seq, label)
                dict_item = dict(
                    meta = dict(
                        header = path_header, seq = seq,
                        label_v1_0 = path_label_v1_0, label_v1_1 = path_label_v1_1,
                        label_v2_0 = path_label_v2_0, label_v2_1 = path_label_v2_1,
                        split = list_dict_split[int(seq)][label]
                    ),
                )
                if self.load_label_in_advance:
                    dict_item = self.get_label(dict_item)
                list_dict_item.append(dict_item)
    
        if split == 'all':
            pass
        else:
            list_dict_item = list(filter(lambda item: item['meta']['split']==split, list_dict_item))

        # Filter unavailable frames (frames wo objects) (only)
        if self.label.remove_0_obj:
            list_dict_item = list(filter(lambda item: item['meta']['num_obj']>0, list_dict_item))

        return list_dict_item
    
    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat('./resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation
    
    def get_label(self, dict_item):
        meta = dict_item['meta']
        temp_key = 'label_' + self.label_version
        path_label = meta[temp_key]
        ver = self.label_version

        f = open(path_label)
        lines = f.readlines()
        f.close()
        list_tuple_objs = []
        deg2rad = np.pi/180.
        
        header = (lines[0]).rstrip('\n')
        try:
            temp_idx, tstamp = header.split(', ')
        except: # line breaking error for v2_0
            _, header_prime, line0 = header.split('*')
            header = '*' + header_prime
            temp_idx, tstamp = header.split(', ')
            # print('* b4: ', lines)
            lines.insert(1, '*'+line0)
            lines[0] = header
            # print('* after: ', lines)
        rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
        tstamp = tstamp.split('=')[1]
        dict_idx = dict(rdr=rdr, ldr64=ldr64, camf=camf,\
                        ldr128=ldr128, camr=camr, tstamp=tstamp)
        if ver == 'v2_0':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                idx_p = int(list_vals[1])
                cls_name = (list_vals[2])
                x = float(list_vals[3])
                y = float(list_vals[4])
                z = float(list_vals[5])
                th = float(list_vals[6])*deg2rad
                l = 2*float(list_vals[7])
                w = 2*float(list_vals[8])
                h = 2*float(list_vals[9])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), 'R'))

        header = dict_item['meta']['header']
        seq = dict_item['meta']['seq']
        path_calib = osp.join(header, seq, 'info_calib', 'calib_radar_lidar.txt')
        dict_path = dict(
            calib = path_calib,
            front = osp.join(header, seq, 'cam-front', f'cam-front_{camf}.png'),
            left = osp.join(header, seq, 'cam-left', f'cam-left_{camr}.png'),
            right = osp.join(header, seq, 'cam-right', f'cam-right_{camr}.png'),
            rear = osp.join(header, seq, 'cam-rear', f'cam-rear_{camr}.png'),
            ldr64 = osp.join(header, seq, 'os2-64', f'os2-64_{ldr64}.pcd'),
            desc = osp.join(header, seq, 'description.txt'),
        )

        onlyR = self.label.onlyR
        consider_cls = self.label.consider_cls
        if consider_cls | onlyR:
            list_temp = []
            for obj in list_tuple_objs:
                cls_name, _, _, avail = obj
                if consider_cls:
                    is_consider, _, _, _ = self.label[cls_name]
                    if not is_consider:
                        continue
                if onlyR:
                    if avail != 'R':
                        continue
                list_temp.append(obj)
            list_tuple_objs = list_temp

        dict_item['meta']['calib'] = self.get_calib_values(path_calib) if self.item.calib else None
        if self.label.calib:
            list_temp = []
            dx, dy, dz = dict_item['meta']['calib']
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                x = x + dx
                y = y + dy
                z = z + dz
                list_temp.append((cls_name, (x, y, z, th, l, w, h), trk, avail))
            list_tuple_objs = list_temp

        if self.label.consider_roi: # after calib
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            check_azimuth_for_rdr = self.roi.check_azimuth_for_rdr
            azimuth_min, azimuth_max = self.roi.azimuth_deg
            rad2deg = 180./np.pi
            temp_list = []
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                azimuth = np.arctan2(y, x)*rad2deg
                if check_azimuth_for_rdr & ((azimuth < azimuth_min) | (azimuth > azimuth_max)):
                    continue
                if (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max) | (z < z_min) | (z > z_max):
                    continue
                temp_list.append(obj)
            list_tuple_objs = temp_list

        num_obj = len(list_tuple_objs)

        dict_item['meta'].update(dict(
            path=dict_path, idx=dict_idx, label=list_tuple_objs, num_obj=num_obj))
        return dict_item
    
    def get_dict_cam_calib_from_yml_all_seqs(self):
        dict_cam_calib_seqs = dict()
        for idx in range(58):
            dict_cam_calib_seqs.update({f'{idx+1}': None})
        
        list_seqs = os.listdir(self.cam_calib.dir)
        list_seqs_int = list(map(lambda x: int(x.split('_')[1]), list_seqs))
        
        list_cams = ['front0', 'front1', 'right0', 'right1', 'rear0', 'rear1', 'left0', 'left1']
        #            'cam_1.yml', 'cam_2.yml', ... , 'cam_8.yml'

        for seq_int in list_seqs_int:
            seq_str = f'{seq_int}'
            dict_cam_calib_seqs[seq_str] = dict()
            dir_seq = osp.join(self.cam_calib.dir, 'seq_' + seq_str.zfill(2))
            list_ymls = sorted(os.listdir(dir_seq))

            assert len(list_ymls) == 8, '* check # of ymls'

            for idx_yml, yml_file in enumerate(list_ymls):
                key_cam = list_cams[idx_yml]
                path_yml = osp.join(dir_seq, yml_file)
                with open(osp.join(path_yml), 'r') as yml_opened:
                    dict_temp = yaml.safe_load(yml_opened)

                if 'img_size_w' not in dict_temp.keys():
                    dict_temp['img_size_w'] = 1280
                    dict_temp['img_size_h'] = 720

                dict_cam_calib_seqs[seq_str].update({key_cam: get_matrices_from_dict_calib(dict_temp)})
                # img_size, intrinsics, distortion, T_ldr2cam
        
        return dict_cam_calib_seqs

    def get_dict_cam_calib_from_yml(self):
        dict_cam_calib = dict()
        dir_cam_calib = self.cam_calib.dir
        list_yml = os.listdir(dir_cam_calib)
        for yml_file_name in list_yml:
            key_name = yml_file_name.split('.')[0].split('_')[1]
            with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
                dict_temp = yaml.safe_load(yml_file)
            dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam
        return dict_cam_calib
    
    def get_dict_cam_calib_from_npy(self, dict_item): # from save_calibration_matrix_in_npy in util_calib.py
        dir_cam_calib = self.cam_calib.dir_npy
        list_npy = os.listdir(dir_cam_calib)

        intrinsic = []
        cam2ldr = []
        ldr2cam = []
        ldr2img = []

        for npy_file_name in list_npy:
            key_name = npy_file_name.split('.')[0].split('_')[1]
            npy_file = osp.join(dir_cam_calib, npy_file_name)
            temp = np.load(npy_file)
            if key_name == 'cam2pix':
                # temp[0,0] *= self.scale_x
                # temp[1,1] *= self.scale_y
                # temp[0,2] *= self.scale_x
                # temp[1,2] *= self.scale_y
                intrinsic.append(temp[:3, :3]) # [3, 3]
            elif key_name == 'ldr2cam':
                ldr2cam.append(temp)
                temp_inv = np.linalg.inv(temp)
                cam2ldr.append(temp_inv) #[4, 4]
        
        for i in range(len(intrinsic)):
            lidar2image = intrinsic[i] @ ldr2cam[i][:3, :4]
            ldr2img.append(lidar2image)
        
        dict_item['camera_intrinsics'] = intrinsic # [3, 3]
        dict_item['camera2lidar'] = cam2ldr # [4, 4]
        dict_item['lidar2image'] = ldr2img # [3, 4]
        
        return dict_item
    
    def get_t_params_all_seqs(self): # from tools/check_camera_kradar
        dir_t_params = self.t_params.dir

        dict_t_params_seq = dict()        
        for t_params_seq in os.listdir(dir_t_params):
            seq = t_params_seq.split('.')[0]
            path_t_params = osp.join(dir_t_params, t_params_seq)
            with open(path_t_params, 'rb') as pickle_file:
                dict_temp_seq = pickle.load(pickle_file)
            dict_t_params_seq[seq] = dict_temp_seq

        self.ref_sensor = self.t_params.ref_sensor

        return dict_t_params_seq
    
    def get_calib_values(self, path_calib):
        f = open(path_calib, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        list_values = [list_calib[1], list_calib[2], self.calib['z_offset']] # X, Y, Z
        return list_values
    
    def get_description(self, dict_item): # ./tools/tag_generator
        f = open(dict_item['meta']['path']['desc'])
        line = f.readline()
        road_type, capture_time, climate = line.split(',')
        dict_desc = {
            'capture_time': capture_time,
            'road_type': road_type,
            'climate': climate,
        }
        f.close()
        dict_item['meta']['desc'] = dict_desc
        
        return dict_item
    ### Setup ###
    
    ### Camera ###
    def get_camera_img(self, dict_item):
        if self.img_load == 0: # cropped
            seq = dict_item['meta']['seq']
            cam_idx = dict_item['meta']['idx']['camf'] # fixed to camf
            for k in self.list_dealt_cams:
                img = Image.open(osp.join(self.img_dir, seq, k, f'{cam_idx}.png'))

                # from matplotlib import pyplot as plt
                # plt.figure()
                # plt.imshow(img)
                # plt.show()  

                if self.cfg_effect is not None:
                    if self.cfg_effect.cam.effect is not None:
                        img_effect = Image.open(self.cfg_effect.cam.path_cropped.get(self.cfg_effect.cam.effect))
                        img_np = np.array(img)
                        img_effect_np = np.array(img_effect)
                        thr = self.cfg_effect.cam.thr
                        img_np[img_effect_np>thr] = img_effect_np[img_effect_np>thr]
                        img = Image.fromarray(img_np)
                        img_np = None
                        img_effect_np = None
                        img_effect = None

                # from matplotlib import pyplot as plt
                # plt.figure()
                # plt.imshow(img)
                # plt.show()

                dict_item[k] = self.compose_img(img)
        elif self.img_load == 1: # undistorted
            seq = dict_item['meta']['seq']
            cam_idx = dict_item['meta']['idx']['camf'] # fixed to camf
            for k in self.list_dealt_cams:
                img = Image.open(osp.join(self.cam_dir.undistorted, seq, k, f'{cam_idx}.png'))
                # TODO
                # Resize & Crop
                dict_item[k] = self.compose_img(img)
        elif self.img_load == 2:
            dict_path = dict_item['meta']['path']
            if self.cam.front0 or self.cam.front1:
                img_front = cv2.imread(dict_path['front'])
                dict_item['front0'] = img_front[:,:1280,:]
                dict_item['front1'] = img_front[:,1280:,:]
            if self.cam.left0 or self.cam.left1:
                img_front = cv2.imread(dict_path['left'])
                dict_item['left0'] = img_front[:,:1280,:]
                dict_item['left1'] = img_front[:,1280:,:]
            if self.cam.right0 or self.cam.left1:
                img_front = cv2.imread(dict_path['right'])
                dict_item['right0'] = img_front[:,:1280,:]
                dict_item['right1'] = img_front[:,1280:,:]
            if self.cam.rear0 or self.cam.rear1:
                img_front = cv2.imread(dict_path['rear'])
                dict_item['rear0'] = img_front[:,:1280,:]
                dict_item['rear1'] = img_front[:,1280:,:]
        
        return dict_item
    
    def get_camera_img_with_key_and_type(self, dict_item, \
            key_cam='front0', img_type='undistorted', is_compose=False):
        seq = dict_item['meta']['seq']
        cam_idx = dict_item['meta']['idx']['camf'] # fixed to camf

        if img_type == 'cropped':
            img = Image.open(osp.join(self.img_dir, seq, key_cam, f'{cam_idx}.png'))
        elif img_type == 'undistorted':
            img = Image.open(osp.join(self.cam_dir.undistorted, seq, key_cam, f'{cam_idx}.png'))

        if is_compose:
            img = self.compose_img(img)
        
        return img
    
    def get_depth_labels(self, dict_item):
        seq = dict_item['meta']['seq']
        cam_idx = dict_item['meta']['idx']['camf'] # fixed to camf
        stride = self.stride_depth_labels
        w_ori, h_ori = self.img_size_depth_labels
        w_stride = int(w_ori/stride)
        h_stride = int(h_ori/stride)
        margin = self.margin_depth_labels
        d_min, d_max = self.d_minmax_depth_labels
        
        for k in self.list_dealt_cams:
            point_depth = np.load(osp.join(self.dir_depth_labels, seq, k, f'{cam_idx}.npy'))
            depth_val = point_depth[:,2]
            
            ind_uv = (np.around(point_depth[:,:2]/stride)).astype(np.int64)
            
            # mask only uv
            # mask_margin = (ind_uv[:,0] >= margin) & (ind_uv[:,0] < w_stride-margin) & \
            #                 (ind_uv[:,1] >= margin) & (ind_uv[:,1] < h_stride-margin)
            
            # mask both uv and d
            mask_margin = (ind_uv[:,0] >= margin) & (ind_uv[:,0] < w_stride-margin) & \
                            (ind_uv[:,1] >= margin) & (ind_uv[:,1] < h_stride-margin) & \
                            (depth_val >= d_min) & (depth_val <= d_max)
            
            ind_u = ind_uv[mask_margin, 0]
            ind_v = ind_uv[mask_margin, 1]
            
            depth_map = np.ones((h_stride, w_stride), dtype=np.float32)*(-100.)
            depth_map[ind_v, ind_u] = depth_val[mask_margin]

            dict_item[k+'_depth'] = depth_map

            # vis
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(depth_map)
            # plt.colorbar()
            # plt.show()
            # print(depth_map.shape)
        
        return dict_item
    
    def get_obj_mask(self, dict_item):
        seq = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr'] # fixed to camf
        dict_item['obj_mask'] = np.load(osp.join(self.dir_obj_mask, seq, f'obj_mask_{rdr_idx}.npy'))

        # vis
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(dict_item['obj_mask'])
        # plt.show()
        
        return dict_item
    
    # def get_img_arrays(self, dict_item):
    #     seq = dict_item['meta']['seq']
    #     cam_idx = dict_item['meta']['idx']['camf'] # fixed to camf
    #     key_cam = 'front0'
    #     num_cam = self.num_img_arrays

    #     for idx_array in range(num_cam):
    #         if idx_array == 0: # t=0 (current img)
    #             img = Image.open(osp.join(self.dir_img_arrays, seq, key_cam, f'{cam_idx}.png'))
                
    #             # TODO: img_processed is little different from img_loaded -> replace 'front0' to loaded one
    #             # vis
    #             # import matplotlib.pyplot as plt
    #             # img = self.compose_img(img)
                
    #             # print(torch.min(img), torch.max(img))
    #             # print(torch.min(dict_item['front0']), torch.max(dict_item['front0']))
    #             # check_equality = (torch.mean(img,0)-torch.mean(dict_item['front0'],0))<0.1
    #             # ind_v, ind_u = torch.where(check_equality==False)

    #             # print('* # pixels equal = ', len(torch.where(check_equality==True)[0]))
    #             # print('* # pixels unequal = ', len(torch.where(check_equality==False)[0]))
                
    #             # plt.subplot(2,1,1)
    #             # plt.imshow(img.permute(1,2,0))
    #             # plt.title('processed')
    #             # plt.subplot(2,1,2)
    #             # plt.imshow(dict_item['front0'].permute(1,2,0))

    #             # plt.scatter(ind_u, ind_v, s=0.5, c='r')
    #             # plt.title('loaded')
    #             # plt.show()

    #             dict_item[key_cam] = self.compose_img(img)
    #         else:
    #             new_idx = int(cam_idx) - self.interval_img_arrays*idx_array
    #             if int(new_idx) < 1:
    #                 new_idx = '00001'
    #             else:
    #                 new_idx = f'{new_idx}'.zfill(5)
    #             img = Image.open(osp.join(self.dir_img_arrays, seq, key_cam, f'{new_idx}.png'))
    #             key_img = key_cam+f'_{idx_array}'
    #             dict_item[key_img] = self.compose_img(img)

    #             # vis
    #             # import matplotlib.pyplot as plt
    #             # img = self.compose_img(img)
    #             # plt.imshow(img.permute(1,2,0))
    #             # plt.show()
        
    #     return dict_item

    def save_undistorted_camera_imgs_w_projected_params(self, dict_args=None, vis=False, save=True):
        func_save_undistorted_camera_imgs_w_projected_params(self, dict_args, vis, save)

    def save_depth_labels_for_cams(self, dict_args=None, vis=False, save=True):
        func_save_depth_labels_for_cams(self, dict_args, vis, save)

    def save_occupied_bev_map(self, dict_args=None):
        func_save_occupied_bev_map(self, dict_args)
    ### Camera ###

    ### LiDAR ###
    def get_ldr64(self, dict_item):
        if self.ldr64.processed: # with attr & calib & roi
            dir_ldr64 = self.ldr64.dir
            seq = dict_item['meta']['seq']
            ldr_idx = dict_item['meta']['idx']['ldr64']
            path_ldr = osp.join(dir_ldr64, seq, f'lpc_{ldr_idx}.npy')
            pc_lidar = np.load(path_ldr)
        else:
            with open(dict_item['meta']['path']['ldr64'], 'r') as f:
                lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
                pc_lidar = [point.split() for point in lines]
                f.close()
            pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)
            
            if self.ldr64.inside_ldr64:
                pc_lidar = pc_lidar[np.where(
                    (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                    (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]

            if self.ldr64.calib:
                n_pts, _ = pc_lidar.shape
                calib_vals = np.array(dict_item['meta']['calib']).reshape(-1,3).repeat(n_pts, axis=0)
                pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        if self.cfg_effect is not None:
            if self.cfg_effect.lid.effect == 'missed':
                deg2rad = np.pi/180.
                min_angle, max_angle = self.cfg_effect.lid.angle
                min_rad = min_angle*deg2rad
                max_rad = max_angle*deg2rad
                azi_lidar = np.arctan2(pc_lidar[:,1],pc_lidar[:,0])
                ind_valid = np.where(((azi_lidar<min_rad) & (azi_lidar>-np.pi)) | \
                                        ((azi_lidar>max_rad) & (azi_lidar<np.pi)))[0]
                pc_lidar = pc_lidar[ind_valid,:]

        if self.cfg_calib_change is not None:
            if self.cfg_calib_change.lid.is_change:
                deg2rad = np.pi/180.
                yaw_angle = self.cfg_calib_change.lid.angle
                yaw_rad = yaw_angle*deg2rad
                dx, dy = self.cfg_calib_change.lid.tra
                
                # First apply translation
                pc_lidar[:,0] += dx
                pc_lidar[:,1] += dy
                
                # Then apply rotation (yaw angle around z-axis)
                rot_matrix = np.array([
                    [np.cos(yaw_rad), -np.sin(yaw_rad)],
                    [np.sin(yaw_rad), np.cos(yaw_rad)]
                ])
                
                # Apply rotation to x,y coordinates
                pc_lidar[:,:2] = np.dot(pc_lidar[:,:2], rot_matrix.T)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,0:3])
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        # vis.add_geometry(coordinate_frame)
        # vis.run()
        # vis.destroy_window()

        dict_item['ldr64'] = pc_lidar

        return dict_item
    
    def get_ldr64_from_path(self, path_ldr64, is_calib=True):
        with open(path_ldr64, 'r') as f:
            lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

        if self.ldr64.inside_ldr64:
            pc_lidar = pc_lidar[np.where(
                (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]
        
        if self.ldr64.calib and is_calib:
            n_pts, _ = pc_lidar.shape
            calib_vals = np.array([-2.54,0.3,0.7]).reshape(-1,3).repeat(n_pts, axis=0)
            pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        return pc_lidar
    
    def save_processed_ldr64_in_npy(self, root_path='/media/donghee/HDD_3/processed_lpc_calib_roi'): # calibration & roi filtering
        for seq_name in range(58):
            seq_folder = osp.join(root_path, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)
        
        for idx_sample in tqdm(range(len(self))):
            dict_item = self.list_dict_item[idx_sample]
            dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item

            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            ldr_idx = dict_meta['idx']['ldr64']

            path_processed_lpc = osp.join(root_path, seq_name, f'lpc_{ldr_idx}.npy')
            if os.path.exists(path_processed_lpc):
                continue

            dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
            dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item

            np.save(path_processed_lpc, dict_item['ldr64'])

            # free memory (Killed error, checked with htop)
            for k in dict_item.keys():
                if k != 'meta':
                    dict_item[k] = None
    ### LiDAR ###
    
    ### 4D Radar ###
    def get_tesseract(self, dict_item):
        seq = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr']
        path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')            
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2)) # DRAE
        dict_item['tesseract'] = arr_tesseract

        return dict_item
    
    def get_cube_polar(self, dict_item, normalizer=1e+13):
        dict_item = self.get_tesseract(dict_item)
        tesseract = dict_item['tesseract'][1:,:,:,:]/normalizer
        cube_pw = np.mean(tesseract, axis=0, keepdims=False)

        # (1) softmax (not used: overflow)    
        # tesseract_exp = np.exp(tesseract)
        # tesseract_exp_sum = np.repeat(np.sum(tesseract_exp, axis=0, keepdims=True), 63, axis=0)
        # tesseract_dist = tesseract_exp/tesseract_exp_sum

        # (2) sum
        tesseract_sum = np.repeat(np.sum(tesseract, axis=0, keepdims=True), 63, axis=0)
        tesseract_dist = tesseract/tesseract_sum

        tesseract_dop = np.reshape(self.arr_doppler[1:], (63,1,1,1)).repeat(256,1).repeat(107,2).repeat(37,3)
        cube_dop = np.sum(tesseract_dist*tesseract_dop, axis=0, keepdims=False)
        
        dict_item['cube_pw_polar'] = cube_pw
        dict_item['cube_dop_cartesian'] = cube_dop

        return dict_item
    
    def save_polar_3d(self, root_path='/media/donghee/kradar/rdr_polar_3d', idx_start=3500, idx_end=4605):
        for seq_name in range(58):
            seq_folder = osp.join(root_path, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)
        
        for idx_sample in tqdm(range(len(self))):
            if (idx_sample < idx_start) or (idx_sample > idx_end):
                continue
            try:
                # dict_item = self.__getitem__(idx_sample)
                # dict_meta = dict_item['meta']
                # seq_name = dict_meta['seq']
                # rdr_idx = dict_meta['idx']['rdr']
                # print('* seq:', seq_name, ', rdr:', rdr_idx)

                dict_item = self.__getitem__(idx_sample)
                
                dict_meta = dict_item['meta']
                seq_name = dict_meta['seq']
                rdr_idx = dict_meta['idx']['rdr']

                path_polar_3d = osp.join(root_path, seq_name, f'polar3d_{rdr_idx}.npy')
                if os.path.exists(path_polar_3d):
                    continue

                dict_item = self.get_cube_polar(dict_item)
                
                cube_pw = dict_item['cube_pw_polar']
                cube_dop = dict_item['cube_dop_cartesian']
                cube_polar = np.stack((cube_pw, cube_dop), axis=0)
                # cube_polar = cube_polar.astype(np.float32)
                # print(cube_polar.shape)
                
                np.save(path_polar_3d, cube_polar)

                # free memory (Killed error, checked with htop)
                for k in dict_item.keys():
                    if k != 'meta':
                        dict_item[k] = None
            except:
                seq = dict_item['meta']['seq']
                rdr_idx = dict_item['meta']['idx']['rdr']
                path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')  
                print(f'* An error happens in {path_tesseract}')
    
    def get_rdr_sparse(self, dict_item):
        if self.rdr_sparse.processed:
            dir_rdr_sparse = self.rdr_sparse.dir
            seq = dict_item['meta']['seq']
            rdr_idx = dict_item['meta']['idx']['rdr']
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'sprdr_{rdr_idx}.npy')
            rdr_sparse = np.load(path_rdr_sparse)
            dict_item['rdr_sparse'] = rdr_sparse
        else: # from cube or tesseract
            # rate = self.rdr_sparse.get('rate', 0.001)
            # dict_item = self.get_portional_rdr_points_from_tesseract(dict_item, rate)
            # dict_item['rdr_sparse'] = dict_item['rdr_pc']
            seq = dict_item['meta']['seq']
            rdr_idx = dict_item['meta']['idx']['rdr']
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'rpc_{rdr_idx}.npy')
            rdr_sparse = np.load(path_rdr_sparse)
            dict_item['rdr_sparse'] = rdr_sparse[:,:4]
        return dict_item

    def get_portional_rdr_points_from_tesseract(self, dict_item, rate=0.001):
        dict_item = self.get_cube_polar(dict_item)

        cube_pw = dict_item['cube_pw_polar'] # Normalized with 1e+13
        cube_dop = dict_item['cube_dop_cartesian'] # Expectation w/ pw dist

        extracted_ind = np.where(cube_pw > np.quantile(cube_pw, 1-rate))
        r_ind, a_ind, e_ind = extracted_ind
        pw = cube_pw[extracted_ind]
        dop = cube_dop[extracted_ind]
        
        r = self.arr_range[r_ind]
        az = self.arr_azimuth[a_ind]
        el = self.arr_elevation[e_ind]

        # Radar polar to General polar coordinate
        az = -az
        el = -el
        
        # For flipped azimuth & elevation angle
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        dict_item['rdr_pc'] = np.stack((x,y,z,pw,dop), axis=1)
        
        return dict_item
    
    def get_rdr_polar_3d(self, dict_item):
        cfg_rdr_polar_3d = self.rdr_polar_3d
        
        if cfg_rdr_polar_3d.processed:
            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            rdr_idx = dict_meta['idx']['rdr']
            path_polar_3d = osp.join(cfg_rdr_polar_3d.dir, seq_name, f'polar3d_{rdr_idx}.npy')
            cube_polar = np.load(path_polar_3d)
        else:
            dict_item = self.get_cube_polar(dict_item)
            cube_pw = dict_item['cube_pw_polar']
            cube_dop = dict_item['cube_dop_cartesian']
            cube_polar = np.stack((cube_pw, cube_dop), axis=0)
        
        if self.get_rdr_polar_3d_in_pc100p:
            pc100p = np.concatenate((self.rdr_polar_3d_xyz, cube_polar), axis=0)
            n_dim, _, _, _ = pc100p.shape
            dict_item['pc100p'] = pc100p.reshape(n_dim,-1).transpose()
        else:
            dict_item['rdr_polar_3d'] = cube_polar
        
        return dict_item # 2, 256, 107, 37 (pw is normalized with 1e+13)
    
    def get_proportional_rdr_points(self, dict_item, rate=0.01, range_wise=False, with_rae_and_ind=False):
        cube_polar = dict_item['rdr_polar_3d']
        # print(cube_polar.shape)
        cube_pw = cube_polar[0,:,:,:]
        cube_dop = cube_polar[1,:,:,:]

        if range_wise:
            n_r, n_a, n_e = cube_pw.shape
            quantile_value = np.quantile(cube_pw, 1-rate, axis=(1,2))
            quantile_value = quantile_value.reshape(n_r,1,1)
            quantile_value = np.repeat(quantile_value, n_a, 1)
            quantile_value = np.repeat(quantile_value, n_e, 2)
            extracted_ind = np.where(cube_pw > quantile_value)
        else:
            extracted_ind = np.where(cube_pw > np.quantile(cube_pw, 1-rate))

        r_ind, a_ind, e_ind = extracted_ind
        pw = cube_pw[extracted_ind]
        dop = cube_dop[extracted_ind]
        
        r = self.arr_range[r_ind]
        az = self.arr_azimuth[a_ind]
        el = self.arr_elevation[e_ind]
        
        # Radar polar to General polar coordinate
        az = -az
        el = -el
        
        # For flipped azimuth & elevation angle
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        if with_rae_and_ind:
            rdr_pc = np.stack((x,y,z,pw,dop,r,az,el,r_ind,a_ind,e_ind), axis=1)
        else:
            rdr_pc = np.stack((x,y,z,pw,dop), axis=1)
        
        dict_item['rdr_pc'] = rdr_pc

        return dict_item
    
    def get_proportional_rdr_points_from_pc100p(self, dict_item, rate=0.01):
        pc100p = dict_item['pc100p']
        quantile_value = np.quantile(pc100p[:,3], 1-rate)
        rdr_pc = pc100p[np.where(pc100p[:,3]>quantile_value)[0],:]
        
        dict_item['rdr_pc'] = rdr_pc
        
        return dict_item
    
    def save_proportional_rdr_pc(self, dict_save=None, root_path='/media/donghee/A4'):#'/media/donghee/kradar/rdr_pc'):
        if dict_save is None:
            dict_save = dict(
                folder_name = 'pc10p-all',
                rate = 0.1,
                range_wise = False,
            )
        
        folder_name = dict_save['folder_name']
        rate = dict_save['rate']
        range_wise = dict_save['range_wise']

        root_path = osp.join(root_path, folder_name)

        for seq_name in range(58):
            seq_folder = osp.join(root_path, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)

        for idx_sample in tqdm(range(len(self))):
            try:
                dict_item = self.__getitem__(idx_sample)
                        
                dict_meta = dict_item['meta']
                seq_name = dict_meta['seq']
                rdr_idx = dict_meta['idx']['rdr']
                
                dict_item = self.get_proportional_rdr_points(dict_item, rate, range_wise, with_rae_and_ind=True)

                path_rpc = osp.join(root_path, seq_name, f'rpc_{rdr_idx}.npy')

                np.save(path_rpc, dict_item['rdr_pc'])

                # free memory (Killed error, checked with htop)
                for k in dict_item.keys():
                    if k != 'meta':
                        dict_item[k] = None
            except:
                seq = dict_item['meta']['seq']
                rdr_idx = dict_item['meta']['idx']['rdr']
                path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')  
                print(f'* An error happens in {path_tesseract}')

    def get_rpcs(self, dict_item):
        if self.rpcs is None:
            return dict_item
        else:
            keys = self.rpcs.keys
            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            rdr_idx = dict_meta['idx']['rdr']

            for temp_key in keys:
                dict_item[temp_key] = np.load(osp.join(self.rpcs.dir, temp_key, seq_name, f'rpc_{rdr_idx}.npy'))

        return dict_item
    ### 4D Radar ###
    
    ### Utils ###
    def filter_roi(self, dict_item):
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
        list_keys = self.roi.keys
        for temp_key in list_keys:
            if temp_key in ['rdr_sparse', 'ldr64']:
                temp_data = dict_item[temp_key]
                temp_data = temp_data[np.where(
                    (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                    (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                    (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
                dict_item[temp_key] = temp_data
            # elif temp_key == 'label': # moved to dict item
        
        return dict_item
    ### Utils ###

    def __len__(self):
        return len(self.list_dict_item)
    
    def __getitem__(self, idx):
        dict_item = self.list_dict_item[idx]
        dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item
        dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
        dict_item = self.get_rdr_polar_3d(dict_item) if self.item['rdr_polar_3d'] else dict_item
        dict_item = self.get_rdr_sparse(dict_item) if self.item['rdr_sparse'] else dict_item
        dict_item = self.get_camera_img(dict_item) if self.item['cam'] else dict_item
        dict_item = self.get_rpcs(dict_item) if self.item['rpcs'] else dict_item
        dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item
        dict_item = self.get_description(dict_item)
        dict_item = self.get_depth_labels(dict_item) if self.load_depth_labels else dict_item
        dict_item = self.get_obj_mask(dict_item) if self.load_obj_mask else dict_item
        # dict_item = self.get_img_arrays(dict_item) if self.load_img_arrays else dict_item
        
        return dict_item
    
    ### Vis ###
    def create_cylinder_mesh(self, radius, p0, p1, color=[1, 0, 0]):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(np.array(p1)-np.array(p0)))
        cylinder.paint_uniform_color(color)
        frame = np.array(p1) - np.array(p0)
        frame /= np.linalg.norm(frame)
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(frame[2]), np.arctan2(-frame[0], frame[1]), 0))
        cylinder.rotate(R, center=[0, 0, 0])
        cylinder.translate((np.array(p0) + np.array(p1)) / 2)
        return cylinder
    
    def draw_3d_box_in_cylinder(self, vis, center, theta, l, w, h, color=[1, 0, 0], radius=0.1, in_cylinder=True):
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]])
        corners = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                            [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])
        corners_rotated = np.dot(corners, R.T) + center
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_rotated)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
        if in_cylinder:
            for line in lines:
                cylinder = self.create_cylinder_mesh(radius, corners_rotated[line[0]], corners_rotated[line[1]], color)
                vis.add_geometry(cylinder)
        else:
            vis.add_geometry(line_set)

    def create_sphere(self, radius=0.2, resolution=30, rgb=[0., 0., 0.], center=[0., 0., 0.]):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        color = np.array(rgb)
        mesh_sphere.vertex_colors = o3d.utility.Vector3dVector([color for _ in range(len(mesh_sphere.vertices))])
        x, y, z = center
        transform = np.identity(4)
        transform[0, 3] = x
        transform[1, 3] = y
        transform[2, 3] = z
        mesh_sphere.transform(transform)
        return mesh_sphere
    
    def vis_in_open3d(self, dict_item, vis_list=['rdr_sparse', 'ldr64', 'label']):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        if 'ldr64' in vis_list:
            pc_lidar = dict_item['ldr64']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
            vis.add_geometry(pcd)

        if 'rdr_sparse' in vis_list:
            rdr_sparse = dict_item['rdr_sparse']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)

        if 'rdr_pc' in vis_list:
            rdr_sparse = dict_item['rdr_pc']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)
        
        if 'label' in vis_list:
            label = dict_item['meta']['label']
            for obj in label:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                consider, logit_idx, rgb, bgr = self.label[cls_name]
                if consider:
                    self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
            vis.run()
            vis.destroy_window()
    ### Vis ###
    
    ### Distribution ###
    def get_distribution_of_label(self, consider_avail=True):
        func_get_distribution_of_label(self, consider_avail)
    ### Distribution ###

    ### Collate ###
    def collate_fn(self, list_batch):
        if None in list_batch:
            print('* Exception error (Dataset): collate fn 0')
            return None
        
        if self.collate_ver == 'v2_0': # gt_boxes (B, M, 8)
            dict_batch = dict()
            
            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []
            dict_batch['gt_boxes'] = []
            
            list_sparse_keys = ['rdr_sparse', 'ldr64', 'pc100p', 'pc30p', 'pc20p', 'pc15p', 'pc10p', 'pc5p', 'pc1p'] # rpcs
            
            max_objs = 0 # for gt_boxes (M)
            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []
                        list_gt_boxes = []
                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                            x, y, z, th, l, w, h = vals
                            list_gt_boxes.append([x, y, z, l, w, h, th, logit_idx])
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                        dict_batch['gt_boxes'].append(list_gt_boxes)
                        max_objs = max(max_objs, dict_item['meta']['num_obj'])
                    elif k in list_sparse_keys:
                        temp_points = dict_item[k]
                        if self.shuffle_points:
                            shuffle_idx = np.random.permutation(temp_points.shape[0])
                            temp_points = temp_points[shuffle_idx]
                        dict_batch[k].append(torch.from_numpy(temp_points).float())
            dict_batch['batch_size'] = batch_idx+1

            batch_size = dict_batch['batch_size']
            gt_boxes = np.zeros((batch_size, max_objs, 8))
            for batch_idx in range(batch_size):
                gt_box = np.array(dict_batch['gt_boxes'][batch_idx])
                gt_boxes[batch_idx,:dict_batch['num_objs'][batch_idx],:] = gt_box
            dict_batch['gt_boxes'] = torch.tensor(gt_boxes, dtype=torch.float32)
            
            for k in list_keys:
                if k in list_sparse_keys:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))
                    
                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)

        elif self.collate_ver == 'v2_1': # camera sensor
            dict_batch = dict()
            
            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []
            dict_batch['gt_boxes'] = []
            
            num_batch = len(list_batch)
            list_sparse_keys = ['rdr_sparse', 'ldr64', 'pc100p', 'pc30p', 'pc20p', 'pc15p', 'pc10p', 'pc5p', 'pc1p'] # rpcs

            # TODO: eye matrix to augmentation
            dict_batch[self.ref_sensor+'_aug_matrix'] = [torch.tensor(np.eye(4), dtype=torch.float32) for _ in range(num_batch)]
            dict_batch[self.ref_sensor+'_aug_matrix'] = torch.stack(dict_batch[self.ref_sensor+'_aug_matrix'], dim=0)

            list_cams = self.list_dealt_cams
            dict_batch['dealt_cams'] = list_cams
            # print(list_cams) # 'front0', 'front1'
            dict_batch['camera_imgs'] = []
            dict_batch['depth_labels'] = []
            dict_batch['camera_intrinsics'] = []
            dict_batch['camera2'+self.ref_sensor] = []
            dict_batch[self.ref_sensor+'2image'] = []
            dict_batch['img_aug_matrix'] = []

            max_objs = 0 # for gt_boxes (M)
            for batch_idx, dict_item in enumerate(list_batch):
                list_camera_imgs = []
                list_depth_labels = []
                list_intrinsics = []
                list_cam2sensor = []
                list_sensor2img = []
                list_aug_matrix = []

                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []
                        list_gt_boxes = []
                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                            x, y, z, th, l, w, h = vals
                            list_gt_boxes.append([x, y, z, l, w, h, th, logit_idx])
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                        dict_batch['gt_boxes'].append(list_gt_boxes)
                        max_objs = max(max_objs, dict_item['meta']['num_obj'])
                    elif k in list_sparse_keys:
                        temp_points = dict_item[k]
                        if self.shuffle_points:
                            shuffle_idx = np.random.permutation(temp_points.shape[0])
                            temp_points = temp_points[shuffle_idx]
                        dict_batch[k].append(torch.from_numpy(temp_points).float())
                    elif k in list_cams:
                        list_camera_imgs.append(v.type(torch.float32))
                        if self.load_depth_labels:
                            list_depth_labels.append(torch.tensor(dict_item[k+'_depth'], dtype=torch.float32))
                        seq = dict_item['meta']['seq']
                        temp_dict_t_params = self.dict_t_params[seq][k]
                        # print(temp_dict_t_params.keys())
                        list_intrinsics.append(torch.tensor(temp_dict_t_params['camera_intrinsics'], dtype=torch.float32))
                        list_cam2sensor.append(torch.tensor(temp_dict_t_params['camera2'+self.ref_sensor], dtype=torch.float32))
                        list_sensor2img.append(torch.tensor(temp_dict_t_params[self.ref_sensor+'2image'], dtype=torch.float32))
                        
                        # TODO: fixed aug matrix to augmentation
                        list_aug_matrix.append(torch.tensor(temp_dict_t_params['img_aug_matrix'], dtype=torch.float32))
                
                # stack along camera
                if len(list_camera_imgs) > 0:
                    dict_batch['camera_imgs'].append(torch.stack(list_camera_imgs, dim=0))
                    if self.load_depth_labels:
                        dict_batch['depth_labels'].append(torch.stack(list_depth_labels, dim=0))
                    dict_batch['camera_intrinsics'].append(torch.stack(list_intrinsics, dim=0))
                    dict_batch['camera2'+self.ref_sensor].append(torch.stack(list_cam2sensor, dim=0))
                    dict_batch[self.ref_sensor+'2image'].append(torch.stack(list_sensor2img, dim=0))
                    dict_batch['img_aug_matrix'].append(torch.stack(list_aug_matrix, dim=0))
            
            # stack along batch & to cuda
            if len(dict_batch['camera_imgs']) > 0:
                dict_batch['camera_imgs'] = torch.stack(dict_batch['camera_imgs'], dim=0)
                # dict_batch['camera_imgs'] = torch.stack(dict_batch['camera_imgs'], dim=0).cuda()

                if self.load_depth_labels:
                    dict_batch['depth_labels'] = torch.stack(dict_batch['depth_labels'], dim=0)
                    # dict_batch['depth_labels'] = torch.stack(dict_batch['depth_labels'], dim=0).cuda()
                
                # cuda in not stereo
                dict_batch['camera_intrinsics'] = torch.stack(dict_batch['camera_intrinsics'], dim=0)
                dict_batch['camera2'+self.ref_sensor] = torch.stack(dict_batch['camera2'+self.ref_sensor], dim=0)
                dict_batch[self.ref_sensor+'2image'] = torch.stack(dict_batch[self.ref_sensor+'2image'], dim=0)
                dict_batch['img_aug_matrix'] = torch.stack(dict_batch['img_aug_matrix'], dim=0)
                # dict_batch[self.ref_sensor+'_aug_matrix'] = dict_batch[self.ref_sensor+'_aug_matrix']

            # print(dict_batch['camera_imgs'].shape)              # b, n, 3, 256, 704
            # print(dict_batch['camera_intrinsics'].shape)        # b, n, 4, 4
            # print(dict_batch['camera2'+self.ref_sensor].shape)  # b, n, 4, 4
            # print(dict_batch[self.ref_sensor+'2image'].shape)   # b, n, 4, 4
            # print(dict_batch['img_aug_matrix'].shape)           # b, n, 4, 4

            dict_batch['batch_size'] = batch_idx+1

            batch_size = dict_batch['batch_size']
            gt_boxes = np.zeros((batch_size, max_objs, 8))
            for batch_idx in range(batch_size):
                gt_box = np.array(dict_batch['gt_boxes'][batch_idx])
                gt_boxes[batch_idx,:dict_batch['num_objs'][batch_idx],:] = gt_box
            dict_batch['gt_boxes'] = torch.tensor(gt_boxes, dtype=torch.float32)
            
            for k in list_keys:
                if k in list_sparse_keys:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))
                    
                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)
            # else:
            #     dict_batch['camera_intrinsics'] = dict_batch['camera_intrinsics'].cuda()
            #     dict_batch['camera2'+self.ref_sensor] = dict_batch['camera2'+self.ref_sensor].cuda()
            #     dict_batch[self.ref_sensor+'2image'] = dict_batch[self.ref_sensor+'2image'].cuda()
            #     dict_batch['img_aug_matrix'] = dict_batch['img_aug_matrix'].cuda()
            #     dict_batch[self.ref_sensor+'_aug_matrix'] = dict_batch[self.ref_sensor+'_aug_matrix'].cuda()

            dict_batch['sensor2image'] = torch.zeros((batch_size,1,4,4), dtype=torch.float32)
            for idx_b in range(dict_batch['batch_size']):
                shaped_intrinsic = dict_batch['camera_intrinsics'][idx_b,:,:,:] # n_cam,4,4
                scale_x = dict_batch['img_aug_matrix'][idx_b,:,0,0]
                scale_y = dict_batch['img_aug_matrix'][idx_b,:,1,1]
                tra_x = dict_batch['img_aug_matrix'][idx_b,:,0,3]
                tra_y = dict_batch['img_aug_matrix'][idx_b,:,1,3]

                shaped_intrinsic[:,0,0] *= scale_x
                shaped_intrinsic[:,1,1] *= scale_y
                shaped_intrinsic[:,0,2] = scale_x*shaped_intrinsic[:,0,2] + tra_x
                shaped_intrinsic[:,1,2] = scale_y*shaped_intrinsic[:,1,2] + tra_y

                if self.cfg_calib_change is not None:
                    if self.cfg_calib_change.cam.is_change:
                        deg2rad = np.pi/180.
                        yaw_angle = self.cfg_calib_change.cam.angle
                        yaw_rad = yaw_angle*deg2rad
                        dx, dy = self.cfg_calib_change.cam.tra
                        
                        # First apply translation
                        dict_batch['camera2'+self.ref_sensor][idx_b,0,0,3] += dx
                        dict_batch['camera2'+self.ref_sensor][idx_b,0,1,3] += dy
                        # dict_batch['camera2'+self.ref_sensor][idx_b,1,0,3] += dx
                        # dict_batch['camera2'+self.ref_sensor][idx_b,1,1,3] += dy
                        
                        # Then apply rotation (yaw angle around z-axis)
                        rot_matrix = torch.tensor([
                            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                            [0, 0, 1]
                        ], dtype=dict_batch['camera2'+self.ref_sensor].dtype, device=dict_batch['camera2'+self.ref_sensor].device)
                        
                        # Apply rotation to x,y coordinates
                        dict_batch['camera2'+self.ref_sensor][idx_b,0,:3,:3] = torch.matmul(rot_matrix, dict_batch['camera2'+self.ref_sensor][idx_b,0,:3,:3])
                        # dict_batch['camera2'+self.ref_sensor][idx_b,1,:3,:3] = torch.matmul(rot_matrix, dict_batch['camera2'+self.ref_sensor][idx_b,1,:3,:3])

                sen2cam0 = torch.inverse(dict_batch['camera2'+self.ref_sensor][idx_b,0,:,:]) # front0 (left) -> reference coord
                # sen2cam1 = torch.inverse(dict_batch['camera2'+self.ref_sensor][idx_b,1,:,:]) # front1 (right)

                # ref as front 0 sensor
                shaped_intrinsic0 = shaped_intrinsic[0,:,:]
                # shaped_intrinsic1 = torch.matmul(shaped_intrinsic[1,:,:], torch.matmul(sen2cam1, dict_batch['camera2'+self.ref_sensor][idx_b,0,:,:]))

                P2 = torch.matmul(shaped_intrinsic0, sen2cam0)
                dict_batch['sensor2image'][idx_b,0,:,:] = P2

                ### [Vis for checking] ###
                # import matplotlib.pyplot as plt
                # def normalize_image(img):
                #     """
                #     Normalize image data to [0, 1] range
                #     """
                #     img_min = np.min(img)
                #     img_max = np.max(img)
                #     if img_max - img_min > 0:
                #         return (img - img_min) / (img_max - img_min)
                #     return img
            
                # lpc = torch.cat([dict_batch['ldr64'][torch.where(dict_batch['batch_indices_ldr64']==idx_b)[0],:3], 
                #     torch.ones_like(dict_batch['ldr64'][torch.where(dict_batch['batch_indices_ldr64']==idx_b)[0],:1])], dim=1)
                
                # img0 = dict_batch['camera_imgs'][idx_b,0,:,:,:].cpu().numpy().transpose(1,2,0)
                # normalized_img0 = normalize_image(img0)
                
                # P2 = torch.matmul(shaped_intrinsic0, sen2cam0)
                # transformed_points0 = torch.matmul(P2, lpc.T).T
                # transformed_points0[:, :2] /= transformed_points0[:, 2:3]

                # points_2d0 = transformed_points0[:,:2]

                # valid_mask0 = (transformed_points0[:, 2] > 0)  # Points in front of camera
                # valid_mask0 = valid_mask0 & (transformed_points0[:, 2] < 72.)  # Points in front of camera
                # valid_mask0 = valid_mask0 & (points_2d0[:, 0] >= 0) & (points_2d0[:, 0] < 703)  # Within image width
                # valid_mask0 = valid_mask0 & (points_2d0[:, 1] >= 0) & (points_2d0[:, 1] < 256)  # Within image height

                # plt.figure(figsize=(20, 10))
                # plt.subplot(2, 1, 1)
                # plt.imshow(normalized_img0)
                # valid_points0 = points_2d0[valid_mask0]
                # ind_u = valid_points0[:, 0].cpu().numpy()
                # ind_v = valid_points0[:, 1].cpu().numpy()
                # val_d = transformed_points0[valid_mask0][:, 2].cpu().numpy()
                # scatter = plt.scatter(ind_u, ind_v, 
                #     c=val_d,
                #     cmap='viridis',
                #     s=20, 
                #     alpha=0.5)
                # plt.title('Camera Image (Left)')
                # plt.colorbar(scatter, label='Depth (m)')

                # for u, v, d in zip(ind_u, ind_v, val_d):
                #     if (d > 70.) and (d < 72.):
                #         plt.text(u, v, 
                #             f'{d}',
                #             fontsize=10,
                #             alpha=0.5,
                #             color='r',
                #             ha='center',
                #             va='center')
                
                # plt.subplot(2, 1, 2)
                # plt.imshow(normalized_img0)
                # valid_depths0 = dict_batch['depth_labels'][idx_b,0,:,:]
                # ind_v, ind_u = torch.where((valid_depths0>0.) & (valid_depths0<72.))
                # valid_depths0 = valid_depths0[ind_v, ind_u]
                # ind_u = ind_u.cpu().numpy()
                # ind_v = ind_v.cpu().numpy()
                # val_d = valid_depths0.cpu().numpy()
                # scatter_depth = plt.scatter(ind_u, ind_v, 
                #     c=val_d,
                #     cmap='viridis',
                #     s=20, 
                #     alpha=0.5)
                # plt.title('Camera Image (Left) from depth label')
                # plt.colorbar(scatter_depth, label='Depth (m)')

                # for u, v, d in zip(ind_u, ind_v, val_d):
                #     if (d > 70.) and (d < 72.):
                #         plt.text(u, v, 
                #             f'{d}',
                #             fontsize=10,
                #             alpha=0.5,
                #             color='r',
                #             ha='center',
                #             va='center')
                # plt.show()

                # print(valid_depths0.shape)
                ### [Vis for checking] ###
            
            # dict_batch['sensor2image'] = dict_batch['sensor2image'].cuda()
        
        ### Obj mask ###
        if self.load_obj_mask:
            dict_batch['obj_mask'] = []
            for idx_batch in range(batch_size):
                dict_batch['obj_mask'].append(torch.tensor(list_batch[idx_batch]['obj_mask']))
            dict_batch['obj_mask'] = torch.stack(dict_batch['obj_mask'])
        ### Obj mask ###

        ### Img arrays ###
        # # print(dict_batch['camera_imgs'].shape)
        # # print(list_batch[0]['front0_1'].shape)
        # if self.load_img_arrays:
        #     key_cam = 'front0' # TODO: generalization for all cams
        #     for idx_array in range(self.num_img_arrays):
        #         if idx_array == 0:
        #             continue
        #         key_img_array = key_cam + f'_{idx_array}'
        #         dict_batch[key_img_array] = torch.stack([torch.tensor(dict_temp[key_img_array]) for dict_temp in list_batch]).cuda()
        #     # print(dict_batch['front0_1'].shape)
        #     # print(dict_batch['front0_2'].shape)
        ### Img arrays ###
        
        dict_batch['pointer'] = list_batch # to release memory
        
        return dict_batch
    ### Collate ###

if __name__ == '__main__':
    kradar_detection = KRadarFusion_v1_0_CalibChange()
    print(len(kradar_detection)) # 34994 for all

    list_batch = [kradar_detection[10000], kradar_detection[15000]]
    dict_batch = kradar_detection.collate_fn(list_batch)

    # kradar_detection.save_occupied_bev_map()
    
    # ## Save camera imgs & calib params ###
    # kradar_detection.save_undistorted_camera_imgs_w_projected_params()
    # ## Save camera imgs & calib params ###
    
    # ## Vis camera calibration ###
    # dict_item = kradar_detection[100]
    # key_name = 'front0'
    # img = dict_item[key_name]
    # ldr64 = kradar_detection.get_ldr64_from_path(dict_item['meta']['path']['ldr64'], is_calib=False) # calibration X
    # seq = dict_item['meta']['seq']
    # show_projected_point_cloud(img, ldr64, kradar_detection.dict_cam_calib[seq][key_name], undistort=True)
    # ## Camera calibration ###

    # ## Save depth_labels for cams ###
    # kradar_detection.save_depth_labels_for_cams(vis=True, save=False)
    # ## Save depth_labels for cams ###

    # dict_item = kradar_detection[1]
    # print(dict_item.keys())
    # print(dict_item['ldr64'].shape)
    # print(np.unique(dict_item['ldr64'][:,3]))
