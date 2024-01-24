'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
* comment: sensor fusion
'''

import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d
import cv2
import yaml

from tqdm import tqdm
from easydict import EasyDict

from scipy.io import loadmat # from matlab

from torch.utils.data import Dataset

try:
    from utils.util_calib import *
except:
    import sys
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util_calib import *

roi = [0,-16,-2,72,16,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/donghee/kradar/dataset'],
        split = ['./resources/split/train.txt', './resources/split/test.txt'],
        revised_label_v1_1 = './tools/revise_label/kradar_revised_label_v1_1',
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
        revised_label_v2_1 = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility',
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
    label_version = 'v2_0', # ['v1_0', 'v1_1', 'v2_0', v2_1']
    item = dict(calib=True, ldr64=True, ldr128=False, rdr=False, rdr_sparse=False, cam=True, rdr_polar_3d=True, rpcs=False),
    calib = dict(z_offset=0.7),
    cam = dict(front0=True, front1=True, left0=False, left1=False, right0=False, right1=False, rear0=False, rear1=False),
    cam_process = dict(origin=(720,1280), cropped=((127,593),(0,1280)), scaled=(256,704), dir='/media/donghee/HDD_3/undistorted_imgs'),
    cam_calib = dict(load=True, dir='./resources/cam_calib/common', dir_npy='./resources/cam_calib/T_npy'),
    ldr64 = dict(processed=False, skip_line=13, n_attr=9, inside_ldr64=True, calib=True,),
    rdr = dict(cube=False,),
    rdr_sparse = dict(processed=True, dir='/media/donghee/kradar/rdr_sparse_data/rtnh_wider_1p_1',),
    rdr_polar_3d = dict(processed=True, dir='/media/donghee/kradar/rdr_polar_3d', in_pc100p=True),
    roi = dict(filter=False, xyz=roi, keys=['ldr64', 'rdr_sparse'], check_azimuth_for_rdr=True, azimuth_deg=[-53,53]),
    rpcs = dict(processed=True, dir='/media/donghee/kradar/rdr_pc', keys=['pc1p', 'pc10p']),
    portion = ['10'], # ['7', '8'],
)

class KRadarDetection_v2_1(Dataset):
    def __init__(self, cfg=None, split='all'):
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
        self.cam_calib = self.cfg.get('cam_calib', None)
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.rdr_polar_3d = self.cfg.get('rdr_polar_3d', None)
        self.rpcs = self.cfg.get('rpcs', None)
        self.roi = self.cfg.roi

        for temp_key in ['cam', 'rdr_polar_3d', 'rpcs']:
            if temp_key not in self.item.keys():
                self.item[temp_key] = False
        
        self.portion = self.cfg.get('portion', None)

        self.list_dict_item = self.load_dict_item(self.cfg.path_data, split)
        if cfg_from_yaml:
            self.cfg.NUM = len(self)
        
        self.collate_ver = self.cfg.get('collate_fn', 'v1_0') # Post-processing

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
        
        self.dict_cam_calib = self.get_dict_cam_calib_from_yml() \
                            if self.cam_calib is not None else None
        
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
        if ver == 'v1_0':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                if len(list_vals) != 11:
                    print('* split err in ', path_label)
                    continue
                idx_p = int(list_vals[1])
                idx_b4 = int(list_vals[2])
                cls_name = list_vals[3]
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p, idx_b4), 'R'))
        elif ver == 'v2_0':
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
        elif ver == 'v2_1':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                avail = list_vals[1]
                idx_p = int(list_vals[2])
                cls_name = (list_vals[3])
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), avail))

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
    
    def save_undistorted_camera_imgs(self, key_cam='front', root_path='/media/donghee/HDD_3/undistorted_imgs'):
        root_path0 = osp.join(root_path, key_cam+'0')
        root_path1 = osp.join(root_path, key_cam+'1')
        
        os.makedirs(root_path0, exist_ok=True)
        os.makedirs(root_path1, exist_ok=True)

        for seq_name in range(58):
            seq_folder0 = osp.join(root_path0, f'{seq_name+1}')
            seq_folder1 = osp.join(root_path1, f'{seq_name+1}')
            os.makedirs(seq_folder0, exist_ok=True)
            os.makedirs(seq_folder1, exist_ok=True)

        key_idx = 'camf' if key_cam == 'front' else 'camr'
        
        list_params0 = self.dict_cam_calib[key_cam+'0']
        list_params1 = self.dict_cam_calib[key_cam+'1']

        # for img 0
        img_size0, intrinsics0, distortion0, T_ldr2cam0 = list_params0
        ncm0, _ = cv2.getOptimalNewCameraMatrix(intrinsics0, distortion0, img_size0, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics0[j,i] = ncm0[j, i]
        map_x0, map_y0 = cv2.initUndistortRectifyMap(intrinsics0, distortion0, None, ncm0, img_size0, cv2.CV_32FC1)

        # for img 1
        img_size1, intrinsics1, distortion1, T_ldr2cam1 = list_params1
        ncm1, _ = cv2.getOptimalNewCameraMatrix(intrinsics1, distortion1, img_size1, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics1[j,i] = ncm1[j, i]
        map_x1, map_y1 = cv2.initUndistortRectifyMap(intrinsics1, distortion1, None, ncm1, img_size1, cv2.CV_32FC1)

        for idx_sample in tqdm(range(len(self))):
            dict_item = self.__getitem__(idx_sample)

            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            cam_idx = dict_meta['idx'][key_idx]

            img_temp = cv2.imread(dict_meta['path']['front'])
            img0 = img_temp[:,:1280,:]
            img1 = img_temp[:,1280:,:]

            img0_undistorted = cv2.remap(img0, map_x0, map_y0, cv2.INTER_LINEAR)
            img1_undistorted = cv2.remap(img1, map_x1, map_y1, cv2.INTER_LINEAR)

            path_img0 = osp.join(root_path0, seq_name, f'cam_{cam_idx}.png')
            path_img1 = osp.join(root_path1, seq_name, f'cam_{cam_idx}.png')
            
            cv2.imwrite(path_img0, img0_undistorted)
            cv2.imwrite(path_img1, img1_undistorted)
    ### Camera ###
    
    ### LiDAR ###
    def get_ldr64(self, dict_item):
        if self.ldr64.processed: # with attr & calib & roi
            pass # TODO
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
            rate = self.rdr_sparse.get('rate', 0.001)
            dict_item = self.get_portional_rdr_points_from_tesseract(dict_item, rate)
            dict_item['rdr_sparse'] = dict_item['rdr_pc']
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
        dict_label = self.label.copy()
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

        for dict_item in tqdm(self.list_dict_item):
            dict_item = self.get_label(dict_item)
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
    ### Distribution ###

    ### Collate ###
    def collate_fn(self, list_batch):
        if None in list_batch:
            print('* Exception error (Dataset): collate fn 0')
            return None
        
        if self.collate_ver == 'v1_0':
            dict_batch = dict()
            
            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []

            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []

                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                    elif k in ['rdr_sparse', 'ldr64']:
                        dict_batch[k].append(torch.from_numpy(dict_item[k]).float())
            dict_batch['batch_size'] = batch_idx+1

            for k in list_keys:
                if k in ['rdr_sparse', 'ldr64']:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))
                    
                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)
        
        elif self.collate_ver == 'v2_0': # gt_boxes (B, M, 8)
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

        dict_batch['pointer'] = list_batch # to release memory

        return dict_batch
    ### Collate ###

if __name__ == '__main__':
    kradar_detection = KRadarDetection_v2_1(split='all')
    print(len(kradar_detection)) # 34994 for all

    dict_item = kradar_detection[0]
    dict_item = kradar_detection.get_dict_cam_calib_from_npy(dict_item)
    
    print(dict_item['camera_intrinsics']) # [3, 3]
    print(dict_item['camera2lidar']) # [4, 4]
    print(dict_item['lidar2image']) # [3, 4]

    print(kradar_detection.dict_cam_calib.keys())
    print(kradar_detection.dict_cam_calib['front0'])
    print(kradar_detection.dict_cam_calib['front1'])
    
    ### Camera calibration ###
    # key_name = 'front0' # ['front0', 'front1', 'left0', 'left1', 'right0', 'right1', 'rear0', 'rear1']
    # img = dict_item[key_name]
    # ldr64 = kradar_detection.get_ldr64_from_path(dict_item['meta']['path']['ldr64'], is_calib=False) # calibration X
    # show_projected_point_cloud(img, ldr64, kradar_detection.dict_cam_calib[key_name], undistort=True)
    # save_calibration_matrix_in_npy(key_name, kradar_detection.dict_cam_calib[key_name], undistort=True)
    ### Camera calibration ###

    ### Save rdr polar 3d ###
    # kradar_detection.save_polar_3d()
    ### Save rdr polar 3d ###

    ### Range-wise proportional rdr points ###
    # dict_item = kradar_detection.get_proportional_rdr_points(dict_item) # pc100p=False
    # dict_item = kradar_detection.get_proportional_rdr_points_from_pc100p(dict_item)
    ### Range-wise proportional rdr points ###

    # print(dict_item['rdr_sparse'].shape)
    # print(dict_item['rdr_pc'].shape)

    ### Vis ###
    # kradar_detection.vis_in_open3d(dict_item, ['ldr64', 'label', 'rdr_pc']) # 'rdr_pc', 'rdr_sparse'
    ### Vis ###

    ### Save rdr pc ###
    # kradar_detection.save_proportional_rdr_pc()
    ### Save rdr pc ###

    ### Save undistorted img ###
    # kradar_detection.save_undistorted_camera_imgs()
    ### Save undistorted img ###

    # kradar_detection.get_distribution_of_label()
