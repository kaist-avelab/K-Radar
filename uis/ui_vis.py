"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2022.05.29
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for vis
"""

# Library
import sys
import os
from PyQt5 import QtGui
import numpy as np
import cv2
import open3d as o3d
import yaml
from easydict import EasyDict
from tqdm import tqdm
import pickle

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# User Library
import datasets
from utils.util_ui_vis import *
from utils.util_geometry import Object3D
from utils.util_config import cfg, cfg_from_yaml_file
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0 as Pipeline

path_ui = '%s/uis/ui_vis.ui' % '.' # cnf.BASE_DIR
class_ui = uic.loadUiType(path_ui)[0]

class MainFrame(QMainWindow, class_ui):
    def __init__(self, cfg):
        super().__init__()
        self.setupUi(self)
        self.cfg = cfg
        self.init_signals()

        ### Global variables ###
        self.kradar = None          # dataset
        self.dict_datum = None      # dict_datum (for GUI)
        self.dict_item = None       # dict_item (for inference)
        self.pred_objects = None
        self.gt_objects = None
        ### Global variables ###

        ### User settings ###
        self.path_prev = None       # for pretrained
        ### User settings ###

        self.setWindowTitle('K-Radar Benchmark')
        qpix_logo = QPixmap('./resources/imgs/logo.png')
        self.label_logo.setPixmap(qpix_logo)
        
    def init_signals(self):
        list_name_fuction = [
            'pushButtonLoad',               # 0
            'pushButtonCalibrate',          # 1
            'pushButtonCameraVis',          # 2
            'pushButtonLidarVis',           # 3
            'pushButtonRadarVis',           # 4
            'pushButtonLoadSubset',         # 5
            'pushButtonLoadConfig',         # 6
            'pushButtonLoadModel',          # 7
            'pushButtonInference',          # 8
            'pushButtonInfCamera',          # 9
            'pushButtonInfLidar',           # 10
            'pushButtonInfRadar',           # 11
            'pushButtonReset',              # 12
            'pushButtonSRTVis',             # 13
            ]
        for i in range(len(list_name_fuction)):
            getattr(self, f'pushButton_{i}').clicked.\
                connect(getattr(self, list_name_fuction[i]))
        self.listWidget_files.itemDoubleClicked.connect(self.listWidget_files_doubleClicked)

    def pushButtonLoad(self):
        ### Change cfg for vis ###
        self.split = 'train' if self.radioButton_training.isChecked() else 'test'
        # self.cfg.DATASET.RDR_CUBE.ROI['x'] = [0, 120]
        # self.cfg.DATASET.RDR_CUBE.ROI['y'] = [-100, 100]
        # self.cfg.DATASET.RDR_CUBE.ROI['z'] = [-50, 50]
        self.cfg.DATASET.RDR_CUBE.ROI['x'] = [0, 98.8]
        self.cfg.DATASET.RDR_CUBE.ROI['y'] = [-40.0, 39.6]
        self.cfg.DATASET.RDR_CUBE.ROI['z'] = [-2, 5.6]
        ### Change cfg for vis ###

        self.kradar = datasets.__all__[self.cfg.DATASET.NAME](cfg=self.cfg, split=self.split)

        ### Add to list widget ###
        self.listWidget_files.clear()

        is_check_dist = self.checkBox_check_dist.isChecked()
        if is_check_dist:
            dict_total_scene_num = {
                'normal':0,'overcast':0,'fog':0,'rain':0,
                'sleet':0,'lightsnow':0,'heavysnow':0,}
            dict_label_dist = {
                'Sedan':0,'Bus or Truck':0,'Motorcycle':0,
                'Bicycle':0,'Pedestrian':0,'Pedestrian Group':0,}
            dict_meter_dist = {
                '10':0,'20':0,'30':0,'40':0,'50':0,
                '60':0,'70':0,'80':0,'90':0,'100':0,
                '110':0,'120':0,'130':0,'140':0,'150':0,}
            list_meter = ['10', '20', '30', '40', '50', '60', '70', \
                '80', '90', '100', '110', '120', '130', '140', '150']
            dict_lwh_dist = {
                'Sedan': [[] for _ in range(3)],'Bus or Truck': [[] for _ in range(3)],
                'Motorcycle': [[] for _ in range(3)],'Bicycle': [[] for _ in range(3)],
                'Pedestrian': [[] for _ in range(3)],'Pedestrian Group': [[] for _ in range(3)],}
            dict_seq_exist_per_class = {
                # 'Sedan': [],'Bus or Truck': [],
                'Motorcycle': [],'Bicycle': [],'Pedestrian': [],'Pedestrian Group': [],}
        
        ### sorting ###
        # We only cares path_label for Vis program: find corresponding data in pipeline for inference by list.idx func
        label_paths = self.kradar.list_path_label.copy()
        label_paths.sort(key=lambda path_label: path_label.split('/')[:-2][-1].zfill(2))
        ### sorting ###

        for idx_label, path_label in tqdm(enumerate(label_paths)): # self.kradar.label_paths)):
            seq_id, radar_idx, lidar_idx, camf_idx = self.kradar.get_data_indices(path_label)
            path_header = path_label.split('/')[:-2]
            seq = path_header[-1]
            path_radar_tesseract = '/'+os.path.join(*path_header, 'radar_tesseract', 'tesseract_'+radar_idx+'.mat')
            path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
            path_radar_bev_img = '/'+os.path.join(*path_header, 'radar_bev_image', 'radar_bev_100_'+radar_idx+'.png')
            path_lidar_bev_img = '/'+os.path.join(*path_header, 'lidar_bev_image', 'lidar_bev_100_'+lidar_idx+'.png')
            path_lidar_pc_64 = '/'+os.path.join(*path_header, 'os2-64', 'os2-64_'+lidar_idx+'.pcd')
            path_lidar_pc_128 = '/'+os.path.join(*path_header, 'os1-128', 'os1-128_'+lidar_idx+'.pcd')
            path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
            path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
            path_desc = '/'+os.path.join(*path_header, 'description.txt')

            dict_datum = dict()

            meta = {
                'path_label': path_label,
                'seq_id': seq_id,
                'rdr_idx': radar_idx,
                'ldr_idx': lidar_idx,
                'camf_idx': camf_idx,
                'path_rdr_tesseract': path_radar_tesseract,
                'path_rdr_cube': path_radar_cube,
                'path_rdr_bev_img': path_radar_bev_img,
                'path_ldr_bev_img': path_lidar_bev_img,
                'path_ldr_pc_64': path_lidar_pc_64,
                'path_ldr_pc_128': path_lidar_pc_128,
                'path_cam_front': path_cam_front,
                'path_calib': path_calib,
                'path_desc': path_desc,
            }

            dict_datum['meta'] = meta

            dict_datum['desc'] = self.kradar.get_description(path_desc)
            cap_time = dict_datum['desc']['capture_time']
            road_type = dict_datum['desc']['road_type']
            climate = dict_datum['desc']['climate']

            if self.kradar.type_coord == 1: # rdr
                dict_datum['calib_info'] = self.kradar.get_calib_info(path_calib)
            else: # ldr
                dict_datum['calib_info'] = None
            
            ### Label ###
            dict_datum['meta']['label'] = self.kradar.get_label_bboxes(path_label, dict_datum['calib_info'])
            ### Label ###

            ### Calculate label distribution ###
            if is_check_dist:
                num_obj = len(dict_datum['meta']['label'])
                dict_total_scene_num[climate] += num_obj
                for tuple_obj in dict_datum['meta']['label']:
                    cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
                    dict_label_dist[cls_name] += 1
                    dist = np.sqrt(x**2+y**2+z**2)
                    # print(dist)
                    dist_key_name = list_meter[int(dist/10)]
                    # print(dist_key_name)
                    dict_meter_dist[dist_key_name] += 1

                    ### Caculate dist 
                    dict_lwh_dist[cls_name][0].append(l)
                    dict_lwh_dist[cls_name][1].append(w)
                    dict_lwh_dist[cls_name][2].append(h)

                    ### Check exist sequence
                    if cls_name in ['Sedan', 'Bus or Truck']:
                        pass
                    else:
                        dict_seq_exist_per_class[cls_name].append(seq) 
            ### Calculate label distribution ###

            with open(path_label, 'r') as f:
                lines = f.readlines()
                f.close()
            tstamp = np.round(float(lines[0].split(',')[1].split('=')[1]), decimals=2)
            temp_item = QListWidgetItem()
            temp_item.setData(1, dict_datum)
            temp_item.setText(str(idx_label) + '. ' + f'seq_{seq}: {tstamp} / {cap_time} / {road_type} / {climate}')
            self.listWidget_files.addItem(temp_item)

        if is_check_dist:
            print('* class distribution')
            for k, v in dict_label_dist.items():
                print(f'{k}: {v}')

            print('* distance distribution')
            for k, v in dict_meter_dist.items():
                print(f'{k}: {v}')

            print('* condition distribution')
            for k, v in dict_total_scene_num.items():
                print(f'{k}: {v}')

            print('* size distribution')
            for k, v in dict_lwh_dist.items():
                print(f'* cls {k}')
                l_m = np.mean(v[0])
                l_std = np.std(v[0])
                w_m = np.mean(v[1])
                w_std = np.std(v[1])
                h_m = np.mean(v[2])
                h_std = np.std(v[2])
                print(f'l={l_m}/{l_std}, w={w_m}/{w_std}, h={h_m}/{h_std}')

            print('* existing sequence')
            for k, v in dict_seq_exist_per_class.items():
                print(f'cls {k}')
                print('seq:', set(v))

    def pushButtonLoadSubset(self):
        self.split = 'train' if self.radioButton_training.isChecked() else 'test'
        self.cfg.DATASET.RDR_CUBE.ROI['x'] = [0, 120]
        self.cfg.DATASET.RDR_CUBE.ROI['y'] = [-100, 100]
        self.cfg.DATASET.RDR_CUBE.ROI['z'] = [-50, 50]

        self.kradar = datasets.__all__[self.cfg.DATASET.NAME](cfg=self.cfg, split=self.split)
        self.kradar.is_roi_check_with_azimuth = False
        self.listWidget_files.clear()
        
        ### sorting ###
        label_paths = self.kradar.label_paths.copy()
        label_paths.sort(key=lambda path_label: path_label.split('/')[:-2][-1].zfill(2))
        ### sorting ###

        arr_idx = np.arange(len(self.kradar.label_paths))
        arr_selected = np.random.choice(arr_idx, self.spinBox_subset.value(), replace=False)
        arr_selected.sort()

        for idx_label, path_label in tqdm(enumerate(label_paths)): # self.kradar.label_paths)):
            if not (idx_label in arr_selected):
                continue

            seq_id, radar_idx, lidar_idx, camf_idx = self.kradar.get_data_indices(path_label)
            path_header = path_label.split('/')[:-2]
            seq = path_header[-1]
            path_radar_tesseract = '/'+os.path.join(*path_header, 'radar_tesseract', 'tesseract_'+radar_idx+'.mat')
            path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
            path_radar_bev_img = '/'+os.path.join(*path_header, 'radar_bev_image', 'radar_bev_100_'+radar_idx+'.png')
            path_lidar_bev_img = '/'+os.path.join(*path_header, 'lidar_bev_image', 'lidar_bev_100_'+lidar_idx+'.png')
            path_lidar_pc_64 = '/'+os.path.join(*path_header, 'os2-64', 'os2-64_'+lidar_idx+'.pcd')
            path_lidar_pc_128 = '/'+os.path.join(*path_header, 'os1-128', 'os1-128_'+lidar_idx+'.pcd')
            path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
            path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
            path_desc = '/'+os.path.join(*path_header, 'description.txt')

            dict_datum = dict()

            meta = {
                'path_label': path_label,
                'seq_id': seq_id,
                'rdr_idx': radar_idx,
                'ldr_idx': lidar_idx,
                'camf_idx': camf_idx,
                'path_rdr_tesseract': path_radar_tesseract,
                'path_rdr_cube': path_radar_cube,
                'path_rdr_bev_img': path_radar_bev_img,
                'path_ldr_bev_img': path_lidar_bev_img,
                'path_ldr_pc_64': path_lidar_pc_64,
                'path_ldr_pc_128': path_lidar_pc_128,
                'path_cam_front': path_cam_front,
                'path_calib': path_calib,
                'path_desc': path_desc,
            }

            dict_datum['meta'] = meta

            ### Process desc (TBD) ###
            dict_datum['desc'] = self.kradar.get_description(path_desc)
            cap_time = dict_datum['desc']['capture_time']
            road_type = dict_datum['desc']['road_type']
            climate = dict_datum['desc']['climate']
            ### Process desc (TBD) ###

            if self.kradar.type_coord == 1: # rdr
                dict_datum['calib_info'] = self.kradar.get_calib_info(path_calib)
            else: # ldr
                dict_datum['calib_info'] = None
            
            ### Label ###
            dict_datum['meta']['label'] = self.kradar.get_label_bboxes(path_label, dict_datum['calib_info'])
            ### Label ###

            with open(path_label, 'r') as f:
                lines = f.readlines()
                f.close()
            tstamp = np.round(float(lines[0].split(',')[1].split('=')[1]), decimals=2)

            temp_item = QListWidgetItem()
            temp_item.setData(1, dict_datum)
            temp_item.setText(str(idx_label) + '. ' + f'seq_{seq}: {tstamp} / {cap_time} / {road_type} / {climate}')
            self.listWidget_files.addItem(temp_item)

    def listWidget_files_doubleClicked(self):
        current_item = self.listWidget_files.currentItem()
        self.textBrowser_logs.append(current_item.data(0) + ' is loaded')
        self.dict_datum_idx = int(current_item.data(0).split('.')[0])
        self.dict_datum = current_item.data(1)

        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        getattr(self, 'label_frontImg').setPixmap(temp_front_img)
        getattr(self, 'label_frontImg_infer').setPixmap(temp_front_img)
        
        labels = self.dict_datum['meta']['label']
        print('-----------------------------------------------------')
        print('* in item: ', labels)
        print('* in item: ', self.dict_datum['meta']['path_label'])

    def pushButtonCalibrate(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return
        
        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        cv_img_ori = cv_img.copy()

        intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(get_list_p_text_edit(self))
        rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

        labels = self.dict_datum['meta']['label']
        list_objs = []
        
        list_line_order = [[0,1], [0,2], [1,3], [2, 3], [0,4], [1,5], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
        for tuple_obj in labels:
            cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
            arr_points = Object3D(x, y, z, l, w, h, theta).corners
            arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
            arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
            list_objs.append(arr_pix)
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = arr_pix[idx_1]
                p2_x, p2_y = arr_pix[idx_2]
                p1_x = int(np.round(p1_x))
                p1_y = int(np.round(p1_y))
                p2_x = int(np.round(p2_x))
                p2_y = int(np.round(p2_y))

                if self.checkBox_color.isChecked():
                    color = self.cfg.VIS.CLASS_BGR[cls_name]
                else:
                    color = (0, 255, 0)
                
                cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=2)
        
        alpha = 0.5
        cv_img = cv2.addWeighted(cv_img, alpha, cv_img_ori, 1 - alpha, 0)
        temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        getattr(self, 'label_frontImg').setPixmap(temp_front_img)

    def pushButtonCameraVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return

        if self.checkBox_thick_alpha.isChecked():
            alpha = self.doubleSpinBox_alpha.value()
            lthick = self.spinBox_lthick.value()
        else:
            alpha = 0.5
            lthick = 2
        
        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        cv_img_ori = cv_img.copy()

        if self.checkBox_bbox.isChecked():
            intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(get_list_p_text_edit(self))
            rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

            labels = self.dict_datum['meta']['label']
            list_objs = []
            
            list_line_order = [[0,1], [0,2], [1,3], [2, 3], [0,4], [1,5], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
            for tuple_obj in labels:
                cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
                arr_points = Object3D(x, y, z, l, w, h, theta).corners
                arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
                arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
                list_objs.append(arr_pix)
                for idx_1, idx_2 in list_line_order:
                    p1_x, p1_y = arr_pix[idx_1]
                    p2_x, p2_y = arr_pix[idx_2]
                    p1_x = int(np.round(p1_x))
                    p1_y = int(np.round(p1_y))
                    p2_x = int(np.round(p2_x))
                    p2_y = int(np.round(p2_y))

                    if self.checkBox_color.isChecked():
                        color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
                    else:
                        color = (0, 255, 0)
                    
                    cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)

        # temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        # getattr(self, 'label_frontImg').setPixmap(temp_front_img)

        cv_img = cv2.addWeighted(cv_img, alpha, cv_img_ori, 1 - alpha, 0)
        cv2.imshow('front_iamge', cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def pushButtonLidarVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return

        self.dict_datum['ldr_pc_64'] = \
            self.kradar.get_pc_lidar(self.dict_datum['meta']['path_ldr_pc_64'], self.dict_datum['calib_info'])
        print(self.dict_datum['calib_info'])

        if self.checkBox_bbox.isChecked():
            bboxes = self.dict_datum['meta']['label']
        else:
            bboxes = None

        if self.checkBox_rdr_pc.isChecked():
            is_with_rdr_pc = True
            cfar_params = self.plainTextEdit_cfar.toPlainText()
            cfar_params = cfar_params.split(',')
            cfar_params = [int(cfar_params[0]), int(cfar_params[1]), float(cfar_params[2])]
            self.dict_datum['rdr_tesseract'] = self.kradar.get_tesseract(self.dict_datum['meta']['path_rdr_tesseract']) 
        else:
            is_with_rdr_pc = False
            cfar_params = [25,8,0.01]
        
        self.kradar.show_lidar_point_cloud(
            self.dict_datum, bboxes, \
            roi_x=[0, 150], roi_y=[-60, 60], roi_z=[-10.0, 10],
            # is_with_rdr_pc=is_with_rdr_pc, cfar_params=cfar_params,
            # roi_x_rdr=[0, 100], roi_y_rdr=[-50, 50], roi_z_rdr=[-2.0, 5],
        )

    def pushButtonRadarVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return

        if self.checkBox_thick_alpha.isChecked():
            alpha = self.doubleSpinBox_alpha.value()
            lthick = self.spinBox_lthick.value()
        else:
            alpha = 0.5
            lthick = 2

        # self.dict_datum = 
        self.dict_datum['rdr_tesseract'] = self.kradar.get_tesseract(self.dict_datum['meta']['path_rdr_tesseract'])

        if self.checkBox_bbox.isChecked():
            bboxes = self.dict_datum['meta']['label']
        else:
            bboxes = None

        print(alpha,lthick)
        print(bboxes)
        self.kradar.show_radar_tensor_bev(self.dict_datum, bboxes, \
            roi_x = [0, 0.4, 80], roi_y = [-60, 0.4, 60], alpha = alpha, lthick = lthick)
        
    def pushButtonLoadConfig(self):
        if self.path_prev is None:
            path_config_folder = './pretrained'
        else:
            path_config_folder = os.path.join(*(self.path_prev.split('/')[:-1]))
        path_config = QFileDialog.getOpenFileName(self, 'Open config file', path_config_folder)[0]
        self.path_prev = path_config
        if not path_config:
            return
        if self.split == 'train':
            mode = 'train'
        elif self.split == 'test':
            mode = 'test'
        self.pline = Pipeline(path_config, mode=mode)
        self.textBrowser_logs.append('* Config is loaded')

    def pushButtonLoadModel(self):
        if self.pline is None:
            self.textBrowser_logs.append('* Config loading is required')
            return
        if self.path_prev is None:
            path_model_folder = './pretrained'
        else:
            path_model_folder = os.path.join(*(self.path_prev.split('/')[:-1]))
        path_model = QFileDialog.getOpenFileName(self, 'Open model file', path_model_folder)[0]
        self.pline.load_dict_model(path_model)
        self.textBrowser_logs.append('* Model is loaded')

    def pushButtonInference(self):
        conf_thr = self.doubleSpinBox_conf_thr.value()

        ### Find correspoinding idx from dict_datum (for GUI) ###
        path_label_from_datum = self.dict_datum['meta']['path_label']
        print(path_label_from_datum)
        
        if self.split == 'train':
            is_train = True
            idx = (self.pline.dataset_train.list_path_label).index(path_label_from_datum)
            print(self.pline.dataset_train.list_path_label[idx])
        elif self.split == 'test':
            is_train = False
            idx = (self.pline.dataset_test.list_path_label).index(path_label_from_datum)
            print(self.pline.dataset_test.list_path_label[idx])

        self.gt_objects, self.pred_objects = self.pline.vis_infer([idx], conf_thr=conf_thr, is_train=is_train)
        print(self.gt_objects, self.pred_objects)

    def pushButtonInfCamera(self):
        if self.gt_objects is None:
            print('* inference is required')
        if self.dict_datum is None:
            print('* select at least one item')
            return

        if self.checkBox_thick_alpha.isChecked():
            alpha = self.doubleSpinBox_alpha.value()
            lthick = self.spinBox_lthick.value()
        else:
            alpha = 0.5
            lthick = 2
        
        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        cv_img_ori = cv_img.copy()

        intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(get_list_p_text_edit(self))
        rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

        labels = self.dict_datum['meta']['label']
        
        list_line_order = [[0,1], [0,2], [1,3], [2, 3], [0,4], [1,5], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
        # for tuple_obj in labels:
        #     cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
        #     arr_points = Object3D(x, y, z, l, w, h, theta).corners
        #     arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
        #     arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
        #     for idx_1, idx_2 in list_line_order:
        #         p1_x, p1_y = arr_pix[idx_1]
        #         p2_x, p2_y = arr_pix[idx_2]
        #         p1_x = int(np.round(p1_x))
        #         p1_y = int(np.round(p1_y))
        #         p2_x = int(np.round(p2_x))
        #         p2_y = int(np.round(p2_y))

        #         if self.checkBox_color.isChecked():
        #             color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
        #         else:
        #             color = (0, 255, 0)
                
        #         cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)

        for tuple_obj in self.gt_objects:
            arr_points = tuple_obj.corners
            arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
            arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = arr_pix[idx_1]
                p2_x, p2_y = arr_pix[idx_2]
                p1_x = int(np.round(p1_x))
                p1_y = int(np.round(p1_y))
                p2_x = int(np.round(p2_x))
                p2_y = int(np.round(p2_y))

                # if self.checkBox_color.isChecked():
                #     color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
                # else:
                #     color = (0, 255, 0)
                
                color = [23,208,253]
                cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)

        for tuple_obj in self.pred_objects:
            arr_points = tuple_obj.corners
            arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
            arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = arr_pix[idx_1]
                p2_x, p2_y = arr_pix[idx_2]
                p1_x = int(np.round(p1_x))
                p1_y = int(np.round(p1_y))
                p2_x = int(np.round(p2_x))
                p2_y = int(np.round(p2_y))

                # if self.checkBox_color.isChecked():
                #     color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
                # else:
                #     color = (0, 255, 0)
                
                color = [0,50,255]
                cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)

        # temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        # getattr(self, 'label_frontImg').setPixmap(temp_front_img)

        cv_img = cv2.addWeighted(cv_img, alpha, cv_img_ori, 1 - alpha, 0)
        cv2.imshow('front_iamge', cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButtonInfLidar(self):
        if self.gt_objects is None:
            print('* inference is required')

        if self.dict_datum is None:
            print('* select at least one item')
            return

        self.dict_datum['ldr_pc_64'] = self.kradar.get_pc_lidar(self.dict_datum['meta']['path_ldr_pc_64'], self.dict_datum['calib_info'])

        pc_lidar = self.dict_datum['ldr_pc_64']

        lines = [ [0,1], [0,2], [1,3], [2,3], [0,4], [1,5],\
                  [2,6], [3,7], [4,5], [4,6], [5,7], [6,7] ]
        
        colors_label = [[1.0,0.815,0.1] for _ in range(len(lines))]
        list_line_set_label = []
        list_line_set_pred = []
        for label_obj in self.gt_objects:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors_label)
            list_line_set_label.append(line_set)
        
        for idx_pred, pred_obj in enumerate(self.pred_objects):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            colors_pred = [[1.,0.2,0.] for _ in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors_pred)
            list_line_set_pred.append(line_set)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])

        o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)

    def pushButtonInfRadar(self):
        if self.gt_objects is None:
            print('* inference is required')

        if self.dict_datum is None:
            print('* select at least one item')
            return

        if self.checkBox_thick_alpha.isChecked():
            alpha = self.doubleSpinBox_alpha.value()
            lthick = self.spinBox_lthick.value()
        else:
            alpha = 0.5
            lthick = 2

        # self.dict_datum = 
        self.dict_datum['rdr_tesseract'] = self.kradar.get_tesseract(self.dict_datum['meta']['path_rdr_tesseract'])

        if self.checkBox_bbox.isChecked():
            bboxes = self.dict_datum['meta']['label']
        else:
            bboxes = None

        print('* alpha: ', alpha, ', lthick: ', lthick)

        rdr_vis_norm = None

        if self.checkBox_hist_stretch.isChecked():
            rdr_vis_norm = 'hist_1.0'
        elif self.checkBox_sat.isChecked():
            alp_val = self.doubleSpinBox_alp.value()
            rdr_vis_norm = f'alp_{alp_val}'

        print('* rdr vis mode: ', rdr_vis_norm)
        self.kradar.show_radar_tensor_bev(self.dict_datum, bboxes, \
            roi_x = [0, 0.4, 80], roi_y = [-60, 0.4, 60], alpha = alpha, lthick = lthick, \
            infer = self.pred_objects, infer_gt = self.gt_objects, norm_img = rdr_vis_norm) # alp_1.0
    
    def pushButtonReset(self):
        with open("./resources/dict_obj.bin", "rb") as f:
            dict_obj = pickle.load(f)

        self.pred_objects = dict_obj["pred"]
        self.gt_objects = dict_obj["gt"]

    def pushButtonSRTVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return

        # dealing cube data
        cfg = self.cfg
        _, _, _, self.kradar.arr_doppler = self.kradar.load_physical_values(is_with_doppler=True)
        # To make BEV -> averaging power
        self.kradar.is_count_minus_1_for_bev = cfg.DATASET.RDR_CUBE.IS_COUNT_MINUS_ONE_FOR_BEV

        # Default ROI for CB (When generating CB from matlab applying interpolation)
        self.kradar.arr_bev_none_minus_1 = None
        self.kradar.arr_z_cb = np.arange(-30, 30, 0.4)
        self.kradar.arr_y_cb = np.arange(-80, 80, 0.4)
        self.kradar.arr_x_cb = np.arange(0, 100, 0.4)

        self.kradar.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI
        if self.kradar.is_consider_roi_rdr_cb:
            self.kradar.consider_roi_cube(cfg.DATASET.RDR_CUBE.ROI)
            if cfg.DATASET.RDR_CUBE.CONSIDER_ROI_ORDER == 'cube -> num':
                self.kradar.consider_roi_order = 1
            elif cfg.DATASET.RDR_CUBE.CONSIDER_ROI_ORDER == 'num -> cube':
                self.kradar.consider_roi_order = 2
            else:
                raise AttributeError('Check consider roi order in cfg')
            if cfg.DATASET.RDR_CUBE.BEV_DIVIDE_WITH == 'bin_z':
                self.kradar.bev_divide_with = 1
            elif cfg.DATASET.RDR_CUBE.BEV_DIVIDE_WITH == 'none_minus_1':
                self.kradar.bev_divide_with = 2
            else:
                raise AttributeError('Check consider bev divide with in cfg')
        self.kradar.is_get_cube_dop = cfg.DATASET.GET_ITEM['rdr_cube_doppler']
        self.kradar.offset_doppler = cfg.DATASET.RDR_CUBE.DOPPLER.OFFSET
        self.kradar.is_dop_another_dir = cfg.DATASET.RDR_CUBE.DOPPLER.IS_ANOTHER_DIR
        self.kradar.dir_dop = cfg.DATASET.DIR.DIR_DOPPLER_CB

        self.dict_datum['rdr_cube'] = self.kradar.get_cube(self.dict_datum['meta']['path_rdr_cube'], mode=1)

        print(self.dict_datum['rdr_cube'].shape)

        if self.checkBox_bbox.isChecked():
            bboxes = self.dict_datum['meta']['label']
        else:
            bboxes = None
        
        import torch
        rdr_cube = torch.from_numpy(self.dict_datum['rdr_cube'])
        rdr_cube_roi = self.cfg.DATASET.RDR_CUBE.ROI
        grid_size = self.cfg.DATASET.RDR_CUBE.GRID_SIZE
        # print(grid_size)
        
        z_min, z_max = rdr_cube_roi['z']
        y_min, y_max = rdr_cube_roi['y']
        x_min, x_max = rdr_cube_roi['x']

        sample_rdr_cube = rdr_cube
        norm_val = float(1e+13)
        sample_rdr_cube = sample_rdr_cube / norm_val
        quantile_rate = 1.0-self.doubleSpinBox_rate.value()
        z_ind, y_ind, x_ind = torch.where(sample_rdr_cube > sample_rdr_cube.quantile(quantile_rate))
        
        power_val = sample_rdr_cube[z_ind, y_ind, x_ind].unsqueeze(-1)

        z_pc_coord = ((z_min + z_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        y_pc_coord = ((y_min + y_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        x_pc_coord = ((x_min + x_ind * grid_size) - grid_size / 2).unsqueeze(-1)

        sparse_rdr_cube = torch.cat((x_pc_coord, y_pc_coord, z_pc_coord, power_val), dim=-1)
        sparse_rdr_cube = sparse_rdr_cube.numpy()

        pc_lidar = []
        with open(self.dict_datum['meta']['path_ldr_pc_64'], 'r') as f:
            lines = [line.rstrip('\n') for line in f][13:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 9)[:, :4]
        # 0.01: filter out missing values
        # pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] > 0.01)].reshape(-1, 4)

        # if self.type_coord == 1: # Rdr coordinate
        #     if calib_info is None:
        #         raise AttributeError('* Exception error (Dataset): Insert calib info!')
            # else:
        calib_info = self.dict_datum['calib_info']
        pc_lidar = np.array(list(map(lambda x: \
            [x[0]+calib_info[0], x[1]+calib_info[1], x[2]+calib_info[2], x[3]],\
            pc_lidar.tolist())))

        self.dict_datum['ldr_pc_64'] = pc_lidar
        pc_lidar = self.dict_datum['ldr_pc_64']

        lpc_roi = self.cfg.DATASET.LPC.ROI
        roi_x = lpc_roi['x']
        roi_y = lpc_roi['y']
        roi_z = lpc_roi['z']
        # ROI filtering
        pc_lidar = pc_lidar[
            np.where(
                (pc_lidar[:, 0] > roi_x[0]) & (pc_lidar[:, 0] < roi_x[1]) &
                (pc_lidar[:, 1] > roi_y[0]) & (pc_lidar[:, 1] < roi_y[1]) &
                (pc_lidar[:, 2] > roi_z[0]) & (pc_lidar[:, 2] < roi_z[1])
            )]

        bboxes = self.dict_datum['meta']['label']

        bboxes_o3d = []
        list_color_bbox = []
        for obj in bboxes:
            cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
            # try item()
            bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta))

            lines = [[0, 1], [2, 3], #[1, 2], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6],[3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_bbox = [self.cfg.VIS.CLASS_RGB[cls_name] for _ in range(len(lines))]
            list_color_bbox.append(colors_bbox)

        line_sets_bbox = []
        for idx_obj, gt_obj in enumerate(bboxes_o3d):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(list_color_bbox[idx_obj])
            line_sets_bbox.append(line_set)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])

        pcd_radar = o3d.geometry.PointCloud()
        pcd_radar.points = o3d.utility.Vector3dVector(sparse_rdr_cube[:, :3])
        pcd_radar.colors = o3d.utility.Vector3dVector(np.repeat(np.array([[0.,0.,0.]]), len(sparse_rdr_cube), axis=0))

        o3d.visualization.draw_geometries([pcd, pcd_radar] + line_sets_bbox)

def startUi(path_cfg):
    f = open(path_cfg, 'r')
    try:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    f.close()

    app = QApplication(sys.argv)
    main_frame = MainFrame(cfg)
    main_frame.show()
    sys.exit(app.exec_())
