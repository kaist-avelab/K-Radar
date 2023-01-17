'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* update: Donghee Paek / Splitting train & test / 2022-03-16
'''

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import sys

import matplotlib.pyplot as plt
import os.path as osp
from glob import glob
from scipy.io import loadmat # from matlab
import pickle

try:
    from utils.util_geometry import *
except:
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from utils.util_geometry import *

class KRadarDataset_v1_1(Dataset):
    def __init__(self, cfg=None, split='train'):
        super().__init__()
        self.cfg = cfg

        ### Load label paths wrt split ###
        # load label paths
        self.split = split # 'train', 'test'
        self.dict_split = self.get_split_dict(self.cfg.DATASET.SPLIT.PATH_SPLIT[split])
        self.label_paths = [] # a list of dic
        for dir_seq in self.cfg.DATASET.SPLIT.LIST_DIR:
            list_seq = os.listdir(dir_seq)
            for seq in list_seq:
                seq_label_paths = sorted(glob(osp.join(dir_seq, seq, 'info_label', '*.txt')))
                seq_label_paths = list(filter(lambda x: (x.split('/')[-1].split('.')[0] in self.dict_split[seq]), seq_label_paths))
                self.label_paths.extend(seq_label_paths)

        # load generated labels (Gaussian confidence) ###
        self.is_use_gen_labels = self.cfg.DATASET.LABEL.IS_USE_PREDEFINED_LABEL
        if self.is_use_gen_labels:
            self.pre_label_dir = self.cfg.DATASET.LABEL.PRE_LABEL_DIR
        self.roi_label = self.cfg.DATASET.LABEL.ROI_CONSIDER_LABEL
        ### Load label paths wrt split ###

        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###
        self.type_coord = self.cfg.DATASET.MODALITY.TYPE_COORD # 1: Radar, 2: Lidar, 3: Camera (TBD)
        self.is_consider_rdr = self.cfg.DATASET.MODALITY.IS_CONSIDER_RDR
        self.is_consider_ldr = self.cfg.DATASET.MODALITY.IS_CONSIDER_LDR
        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###

        ### Radar ###
        # dealing cube data
        self.is_count_minus_1_for_bev = False
        self.arr_bev_none_minus_1 = None
        # load physical values
        self.arr_range, self.arr_azimuth, self.arr_elevation = self.load_physical_values()
        # consider roi
        self.is_consider_roi_rdr = cfg.DATASET.RDR.IS_CONSIDER_ROI_RDR
        if self.is_consider_roi_rdr:
            self.consider_roi_rdr(cfg.DATASET.RDR.RDR_POLAR_ROI)
        ### Radar ###
        
        ### Lidar ###
        # (TBD)
        ### Lidar ###

        ### Camera ###
        # (TBD)
        ### Camera ###

    def get_split_dict(self, path_split):
        f = open(path_split, 'r')
        lines = f.readlines()
        f.close

        dict_seq = dict()
        for line in lines:
            seq = line.split(',')[0]
            label = line.split(',')[1].split('.')[0]

            if not (seq in list(dict_seq.keys())):
                dict_seq[seq] = []
            
            dict_seq[seq].append(label)

        return dict_seq

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

    def consider_roi_rdr(self, roi_polar, is_reflect_to_cfg=True):
        self.list_roi_idx = [0, len(self.arr_range)-1, \
            0, len(self.arr_azimuth)-1, 0, len(self.arr_elevation)-1]

        idx_attr = 0
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi

        for k, v in roi_polar.items():
            if v is not None:
                min_max = (np.array(v)*deg2rad).tolist() if idx_attr > 0 else v
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                self.list_roi_idx[idx_attr*2] = idx_min
                self.list_roi_idx[idx_attr*2+1] = idx_max
                
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new =  (np.array(v_new)*rad2deg) if idx_attr > 0 else v_new
                    self.cfg.DATASET.RDR.RDR_POLAR_ROI[k] = v_new
            idx_attr += 1

        # print(self.cfg.DATASET.RDR.RDR_POLAR_ROI)
        # print(self.arr_range)
        # print(self.arr_azimuth*rad2deg)
    
    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        
        return arr[idx_min:idx_max+1], idx_min, idx_max

    def get_calib_info(self, path_calib, is_z_offset_from_cfg=True):
        '''
        * return: [X, Y, Z]
        * if you want to get frame difference, get list_calib[0]
        '''
        with open(path_calib) as f:
            lines = f.readlines()
            f.close()
            
        try:
            list_calib = list(map(lambda x: float(x), lines[1].split(',')))
            # list_calib[0] # frame difference
            list_values = [list_calib[1], list_calib[2]] # X, Y
            
            if is_z_offset_from_cfg:
                list_values.append(self.cfg.DATASET.Z_OFFSET) # Z
            else:
                list_values.append(0.)

            return np.array(list_values)
        except:
            raise FileNotFoundError('no calib info')

    def get_tuple_object(self, line, calib_info, is_heading_in_rad=True):
        '''
        * in : e.g., '*, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> One Example
        * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
        * out: tuple ('Sedan', idx_cls, [x, y, z, theta, l, w, h], idx_obj)
        *       None if idx_cls == -1 or header != '*'
        '''
        list_values = line.split(',')

        if list_values[0] != '*':
            return None

        offset = 0
        if(len(list_values)) == 11:
            offset = 1
        cls_name = list_values[2+offset][1:]

        idx_cls = self.cfg.DATASET.CLASS_ID[cls_name]

        if idx_cls == -1: # Considering None as -1
            return None

        idx_obj = int(list_values[1+offset])
        x = float(list_values[3+offset])
        y = float(list_values[4+offset])
        z = float(list_values[5+offset])
        theta = float(list_values[6+offset])
        if is_heading_in_rad:
            theta = theta*np.pi/180.
        l = 2*float(list_values[7+offset])
        w = 2*float(list_values[8+offset])
        h = 2*float(list_values[9+offset])

        if self.type_coord == 1: # radar coord
            # print('calib_info = ', calib_info)
            x = x+calib_info[0]
            y = y+calib_info[1]
            z = z+calib_info[2]

        # if the label is in roi
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi_label
        if ((x > x_min) and (x < x_max) and \
            (y > y_min) and (y < y_max) and \
            (z > z_min) and (z < z_max)):
            return (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
        else:
            return None

    def get_label_bboxes(self, path_label, calib_info):
        with open(path_label, 'r') as f:
            lines = f.readlines()
            f.close()
        line_objects = lines[1:]
        list_objects = []

        # print(dict_temp['meta']['path_label'])
        for line in line_objects:
            temp_tuple = self.get_tuple_object(line, calib_info)
            if temp_tuple is not None:
                list_objects.append(temp_tuple)

        return list_objects
    
    def __len__(self):
            return len(self.label_paths)
    
    def __getitem__(self, idx):
        # t1 = time.time()
        
        path_label = self.label_paths[idx]

        path_header = path_label.split('/')[:-2]
        path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
        dic = {}

        if self.type_coord == 1: # rdr
            dic['calib_info'] = self.get_calib_info(path_calib)
        else: # ldr
            dic['calib_info'] = None
        
        ### Label ###
        dic['label'] = self.get_label_bboxes(path_label, dic['calib_info'])
        ### Label ###

        return dic

if __name__ == '__main__':
    ### temp library ###
    import yaml
    from easydict import EasyDict
    path_cfg = './configs/cfg_total_v1/cfg_total_v1_0/ResNext4D.yml'
    f = open(path_cfg, 'r')
    try:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    f.close()
    
    dataset = KRadarDataset_v1_1(cfg=cfg, split='train')
    
    list_cls = ['Sedan', 'Bus or Truck']
    list_obj = [[] for _ in range(len(list_cls))]

    for idx_datum in range(len(dataset)):
        for temp_bbox in dataset[idx_datum]['label']:
            name_cls = temp_bbox[0]
            if name_cls in list_cls:
                idx_cls = list_cls.index(name_cls)
                list_obj[idx_cls].append(temp_bbox)

    # print(len(list_obj[0]))
    # print(len(list_obj[1]))
    
    for idx_cls in range(len(list_cls)):
        list_obj_single = list_obj[idx_cls]
        name_obj = list_cls[idx_cls]

        list_obj_z = [obj_single[2][2] for obj_single in list_obj_single]

        list_obj_l = [obj_single[2][4] for obj_single in list_obj_single]
        list_obj_w = [obj_single[2][5] for obj_single in list_obj_single]
        list_obj_h = [obj_single[2][6] for obj_single in list_obj_single]

        mean_z = np.mean(np.array(list_obj_z))

        mean_l = np.mean(np.array(list_obj_l))
        mean_w = np.mean(np.array(list_obj_w))
        mean_h = np.mean(np.array(list_obj_h))

        # plt.hist(list_obj_z, label=f'{name_obj}, l={mean_z}')

        # plt.hist(list_obj_l, label=f'{name_obj}, l={mean_l}')
        # plt.hist(list_obj_w, label=f'{name_obj}, l={mean_w}')
        # plt.hist(list_obj_h, label=f'{name_obj}, l={mean_h}')
    
    # plt.legend()
    # plt.show()
