"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.10.07
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for calibration
"""

# Library
import sys
import os
import yaml
from PyQt5 import QtGui
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# User Library
import configs.config_general as cnf
import configs.config_ui as cnf_ui

from utils.util_ui_labeling import *
from utils.util_ui_labeling import BoundingBox
from utils.util_point_cloud import *

path_ui = '%s/uis/ui_labeling.ui' % cnf.BASE_DIR
class_ui = uic.loadUiType(path_ui)[0]

savepath = './resources/cam_calib/common'
loadpath = './resources/cam_calib/common'

class MainFrame(QMainWindow, class_ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.initGlobalVariables()
        self.initSignals()
        # self.initThread()
        self.initDetails()

    def initGlobalVariables(self):
        # Initialize global variables...
        self.state_global = 0
        self.state_local = 0
        self.idx_log = 0

        # Buffer
        self.idx_cls_bbox = 0
        self.list_cls_bbox = [] # Class BBox
        self.cv_img = None      # Color Img

        self.list_dict_lidar = None
        self.list_dict_radar = None
        self.dict_lidar = None
        self.dict_radar = None

        self.calib_base = cnf_ui.CALIB      # x, y, yaw / Translation -> Rotation
        self.calib_offset = [0., 0., 0.]    # [m, m, deg]
        
        self.path_seq_dir = cnf_ui.PATH_SEQ
        self.path_lidar = None
        self.path_radar = None

        self.seq_name = None

        self.is_enable_right_button = False
        self.is_start_z_labeling = False

        self.diff_frame = self.spinBoxDelay.value()
        self.range_vis = self.horizontalSliderVisRange.value()
        self.str_time = '1631100540117923'
        self.name_labeler = None
        self.idx_interval_log = 0
        self.idx_file_log = None
        # self.plainTextEditLabels.appendPlainText(\
        #     f'* Time: {self.str_time} / Object Num, Class, (Center [m]), Azimuth [deg], (Length [m]), 4 decimal points, # = 2D info, * = 3D info')

    def initSignals(self):
        # Initialize signals...
        list_name_fuction = [ \
            'pushButtonLabeling',           # 0
            'pushButtonSaveLabel',          # 1
            'pushButtonRotateCcw',          # 2
            'pushButtonLoadLabel',          # 3
            'pushButtonPcOpen3d',           # 4
            'pushButtonUpdateBboxToImg',    # 5
            'pushButtonBoundaryUp',         # 6
            'pushButtonCenterUp',           # 7
            'pushButtonCenterDown',         # 8
            'pushButtonBoundaryDown',       # 9
            'pushButtonStart3dLabeling',    # 10
            'pushButtonVisSensorSuit',      # 11
            'pushButtonPcOpenCropped',      # 12
            'pushButtonModifyBoxUp',        # 13
            'pushButtonModifyBoxDown',      # 14
            'pushButtonModifyBoxLeft',      # 15
            'pushButtonModifyBoxRight',     # 16
            'pushButtonModifyBoxLxUp',      # 17
            'pushButtonModifyBoxLxDown',    # 18
            'pushButtonModifyBoxLyDown',    # 19
            'pushButtonModifyBoxLyUp',      # 20
            'pushButtonBackToCalib',        # 21
            'pushButtonRotateCw',           # 22
            'pushButtonFrontImg',           # 23
            'pushButtonLefImg',             # 24
            'pushButtonRightImg',           # 25
            'pushButtonRearImg',            # 26
            'pushButtonVisualizeBBox',      # 27
            'pushButtonLoadLabelPrev',]     # 28
        list_name_function_calib = [ \
            'pushButtonCalibUp',            # 0
            'pushButtonCalibLeft',          # 1
            'pushButtonCalibRight',         # 2
            'pushButtonCalibDown',          # 3
            'pushButtonCalibCcw',           # 4
            'pushButtonCalibCw',            # 5
            'pushButtonCalibBasePath',      # 6
            'pushButtonCalibLoad',          # 7
            'pushButtonCalibGoToLabel',     # 8
            'pushButtonCalibSaveFixed',     # 9
            'pushButtonCalibSaveDialog',    # 10
            'pushButtonCalibUpdateBev',     # 11
            'pushButtonUpdateBaseParams',   # 12
            'pushButtonCalibPlayBack',      # 13
            'pushButtonCalibGenerateText',  # 14
            'pushButtonSetFontSize',        # 15
            'pushButtonSetFrameDiff',       # 16
            'pushButtonNameLabeler',        # 17
            'pushButtonGoToLcCalib',        # 18
            'pushButtonShowTrackInfo',]     # 19
        list_name_function_lc = [ \
            'pushButtonLcBackToRlCalib',    # 0
            'pushButtonLcShowSensorSuite',  # 1
            'pushButtonLcInitValue',        # 2
            'pushButtonLcShowCalib',        # 3
            'pushButtonLcShowRoiPc',        # 4
             # camera selection (stereo) #
            'pushButtonLcFrontLeft',        # 5
            'pushButtonLcFrontRight',       # 6
            'pushButtonLcRearLeft',         # 7
            'pushButtonLcRearRight',        # 8
            'pushButtonLcLeftLeft',         # 9
            'pushButtonLcLeftRight',        # 10
            'pushButtonLcRightLeft',        # 11
            'pushButtonLcRightRight',       # 12
            'pushButtonLcSaveParams',       # 13
            'pushButtonLcChangeSequence',   # 14
            'pushButtonLcLoadParams',       # 15
            ]       
        list_name_itemwidget = [ \
            'listWidgetSequence',
            'listWidgetLidar',
            'listWidgetRadar',]

        for i in range(len(list_name_fuction)):
            getattr(self, f'pushButton_{i}').clicked.\
                connect(getattr(self, list_name_fuction[i]))
        
        for i in range(len(list_name_function_calib)):
            getattr(self, f'pushButtonCalib_{i}').clicked.\
                connect(getattr(self, list_name_function_calib[i]))

        for i in range(len(list_name_function_lc)):
            getattr(self, f'pushButtonLcCalib_{i}').clicked.\
                connect(getattr(self, list_name_function_lc[i]))
        
        for i in range(len(list_name_itemwidget)):
            getattr(self, list_name_itemwidget[i]).itemDoubleClicked.\
                connect(getattr(self, f'itemDoubleClicked_{list_name_itemwidget[i]}'))

        self.horizontalSliderVisRange.valueChanged.connect(\
                self.horizontalSliderVisRangeChanged)
        self.checkBox_7.stateChanged.connect(self.setLabelingRadarView)
        self.spinBoxLcCalib_0.valueChanged.connect(self.idx_changed)

    def initDetails(self):
        self.stackedWidget.setCurrentIndex(1)
        # Initialize fonts
        updateUiDetails(self, cnf_ui.FONT_SIZE)

        # Initialize imgs
        self.cv_img = cv2.imread(os.path.join(\
            cnf.BASE_DIR, 'resources', 'imgs', 'example_bev.png'))
        self.updateBevImage(self.cv_img)
        cv_img = cv2.imread(os.path.join(\
            cnf.BASE_DIR, 'resources', 'imgs', 'example_cam.png'))
        self.updateCamImage(cv_img)
        self.checkBox_0.setChecked(False)

    def mousePressEvent(self, e):
        if self.state_global == cnf_ui.SG_NORMAL:
            return

        x = e.x()
        y = e.y()

        if e.buttons() & Qt.LeftButton:
            type_bt = cnf_ui.BT_LEFT
        elif e.buttons() & Qt.RightButton:
            type_bt = cnf_ui.BT_RIGHT
        elif e.buttons() & Qt.MiddleButton:
            type_bt = cnf_ui.BT_MIDDLE

        if not get_is_point_in_bev_img(x, y):
            return

        if self.state_global == cnf_ui.SG_START_LABELING:
            self.state_local = process_bbox_wrt_state_local(self, self.state_local, x, y, type_bt)

    def setLabelingRadarView(self): # Lidar coordinate
        if self.checkBox_7.isChecked():
            if self.range_vis > 100:
                self.range_vis = 100 # not more than 100, 110 does not exists.
            self.cv_img = get_bev_img_wrt_vis_range_radar(self, self.range_vis)
        else:
            self.cv_img = get_bev_img_wrt_vis_range(self, self.range_vis)
        self.updateBevImage(self.cv_img)
        self.backupBevImage('global')

    def wheelEvent(self, e):
        if self.checkBox_0.isChecked():
            if e.angleDelta().y() > 0:
                # print('wheel up')
                temp_value = self.horizontalSliderVisRange.value()
                if temp_value < 110:
                    self.horizontalSliderVisRange.setValue(temp_value+1)
                elif temp_value > 110:
                    self.horizontalSliderVisRange.setValue(110)

            elif e.angleDelta().y() < 0:
                # print('wheel down')
                temp_value = self.horizontalSliderVisRange.value()
                if temp_value > 10:
                    self.horizontalSliderVisRange.setValue(temp_value-1)
                elif temp_value < 10:
                    self.horizontalSliderVisRange.setValue(10)

    def backupBevImage(self, type_backup = 'local'):
        if type_backup == 'local':
            cv2.imwrite(cnf_ui.PATH_IMG_L, self.cv_img)
        elif type_backup == 'global':
            cv2.imwrite(cnf_ui.PATH_IMG_G, self.cv_img)

    def updateBevImage(self, cv_img = None, type_load = 'local', is_resize = False):
        if cv_img is None:
            if type_load == 'local':
                cv_img = cv2.imread(cnf_ui.PATH_IMG_L)
            elif type_load == 'global':
                cv_img = cv2.imread(cnf_ui.PATH_IMG_G)

        if is_resize:
            cv_img =  cv2.resize(cv_img, \
                (cnf_ui.W_BEV, cnf_ui.H_BEV), interpolation=cv2.INTER_LINEAR)

        self.cv_img = cv_img
        self.labelBev.setPixmap(get_q_pixmap_from_cv_img(cv_img))

    def updateCamImage(self, cv_img = None):
        cv_img = cv2.resize(cv_img, \
            (cnf_ui.W_CAM, cnf_ui.H_CAM), interpolation=cv2.INTER_LINEAR)
        self.labelCam.setPixmap(get_q_pixmap_from_cv_img(cv_img))

    def addLogs(self, logs, is_save_log=True, save_interval=50):
        # in
        #   e.g., 'hi' or ['hi', 'hey']
        if isinstance(logs, list):
            for i in range(len(logs)):
                temp_log = logs[i]
                temp_header = f'{self.idx_log}'.zfill(3)
                self.textBrowserLogs.append(f'{temp_header}: {temp_log}')
                self.idx_log += 1
        else:
            temp_header = f'{self.idx_log}'.zfill(3)
            self.textBrowserLogs.append(f'{temp_header}: {logs}')
            self.idx_log += 1

        if is_save_log:
            self.idx_interval_log += 1
            if self.idx_interval_log == save_interval:
                if self.idx_file_log is None:
                    self.idx_file_log = len(os.listdir(os.path.join(self.path_seq_dir, self.seq_name, 'info_frames')))
                f = open(os.path.join(self.path_seq_dir, self.seq_name, 'info_frames', f'log_{self.idx_file_log}.txt'), 'w')
                now_time = get_now_time_string()
                txt_log = f'labeler = {self.name_labeler}\n' + f'time = {now_time}\n' + self.textBrowserLogs.toPlainText()
                f.write(txt_log)
                f.close()
                self.idx_interval_log = 0

    def pushButtonLabeling(self):
        self.addLogs(['Start labeling ...', 'Click center point ...'])
        self.state_global = cnf_ui.SG_START_LABELING
        self.state_local = cnf_ui.SL_START_LABELING

    def pushButtonPcOpen3d(self):
        # path_pcd = os.path.join(cnf_ui.BASE_DIR, 'data', 'example', f'pc_{self.str_time}.pcd')
        path_pcd = self.dict_lidar['pc']
        print(f'path_pcd = {path_pcd}')
        pcd = o3d.io.read_point_cloud(path_pcd)
        o3d.visualization.draw_geometries([pcd])

        ### Cropping Function with BBox ###
        # https://github.com/isl-org/Open3D/issues/1410

    def pushButtonPcOpenCropped(self):
        pc_os64 = get_filtered_point_cloud_from_plain_text(self)

        if pc_os64 is None:
            self.addLogs('Create a label first!')
            return
        
        pcd = o3d.geometry.PointCloud()
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pcd.points = o3d.utility.Vector3dVector(pc_os64['values'][:,:3])
        # pcd = o3d.io.read_point_cloud(path_pcd)
        o3d.visualization.draw_geometries([pcd])#,frame])

    def pushButtonUpdateBboxToImg(self):
        is_update_plain_text_edit = True

        plain_text = self.plainTextEditLabels.toPlainText()
        list_dict_bbox = get_list_dict_by_processing_plain_text(plain_text)

        plain_text_update = ''
        if is_update_plain_text_edit:
            radar_idx = self.dict_radar['idx_str']
            lidar_idx = self.dict_lidar['idx_str']
            camera_idx = self.dict_lidar['front_img'].split(cnf_ui.SPLITTER)[-1].split('.')[0].split('_')[-1]
            time_string = self.dict_lidar['timestamp_pc']
            if not (self.dict_lidar['idx_prev_str'] is None):
                lidar_idx_prev = self.dict_lidar['idx_prev_str']
            else:
                lidar_idx_prev = -1
            
            plain_text_update += f'* radar idx: {radar_idx}, lidar idx: {lidar_idx}, camera idx: {camera_idx}, time: {time_string}, prev lidar idx: {lidar_idx_prev}\n'
        
        self.list_cls_bbox.clear()
        self.idx_cls_bbox = 0
        cv_img = cv2.imread(cnf_ui.PATH_IMG_G)
        for idx, dict_bbox in enumerate(list_dict_bbox):
            temp_bbox = BoundingBox()
            
            if dict_bbox['type'] == '#':
                list_infos = [dict_bbox['x'], dict_bbox['y'], \
                    dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l']]
                temp_bbox.set_2d_bbox_infos_in_meter(list_infos)
            elif dict_bbox['type'] == '*':
                list_infos = [dict_bbox['x'], dict_bbox['y'], dict_bbox['z'], \
                    dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l'], dict_bbox['z_l']]
                temp_bbox.set_3d_bbox_infos_in_meter(list_infos)
            
            idx_prev = dict_bbox['idx_bbox_prev']

            if is_update_plain_text_edit:
                plain_text_update += get_statement_bbox(list_infos, dict_bbox['cls'], idx, idx_prev)
                plain_text_update += '\n'

            temp_bbox.set_pix_from_2d_bbox_infos(self.range_vis)
            pts = temp_bbox.get_bounding_box_4_points(is_index=True)
            _, color = self.getClsNameAndColor(dict_bbox['cls'])
            x_cen = int(np.round(temp_bbox.x_pix))
            y_cen = int(np.round(temp_bbox.y_pix))
            cv_img = draw_bbox_outline(cv_img, pts, color, \
                is_with_azi=True, cen_to_front=[ x_cen, y_cen,
                                                 int(np.round(temp_bbox.x_f_pix)),
                                                 int(np.round(temp_bbox.y_f_pix))])
            cv2.putText(cv_img, f'{idx}', (x_cen, y_cen), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1, cv2.LINE_AA)
            self.updateBevImage(cv_img)
            self.list_cls_bbox.append(temp_bbox)
            self.idx_cls_bbox += 1

        if is_update_plain_text_edit:
            self.plainTextEditLabels.clear()
            plain_text_update = plain_text_update[:-1]
            self.plainTextEditLabels.setPlainText(plain_text_update)

        return

    def horizontalSliderVisRangeChanged(self):
        self.range_vis = self.horizontalSliderVisRange.value()
        
        # print(f'in callback {self.range_vis}')
        if self.checkBox_7.isChecked():
            if self.range_vis > 100:
                self.range_vis = 100 # not more than 100, 110 does not exists.
                self.horizontalSliderVisRange.setValue(100)
            self.cv_img = get_bev_img_wrt_vis_range_radar(self, self.range_vis)
        else:
            self.cv_img = get_bev_img_wrt_vis_range(self, self.range_vis)
        self.updateBevImage(self.cv_img)
        self.backupBevImage('global')

    def getClsNameAndColor(self, cls_name=None):
        # if cls_name is None
        if cls_name is None:
            for i in range(7):
                if getattr(self, f'radioButton_{i}').isChecked():
                    return cnf_ui.LIST_CLS_NAME[i], cnf_ui.LIST_CLS_COLOR[i]
        else:
            idx = cnf_ui.LIST_CLS_NAME.index(cls_name)
            return cnf_ui.LIST_CLS_NAME[idx], cnf_ui.LIST_CLS_COLOR[idx]
    
    def pushButtonStart3dLabeling(self):
        # try:
        pc_os64 = get_filtered_point_cloud_from_plain_text(self)
        if pc_os64 is None:
            if self.checkBox_7.isChecked():
                img_h, img_w = cnf_ui.IMG_SIZE_YZ
                img_bev_f = np.full((img_h,img_w,3), 255, dtype=np.uint8)
                img_h, img_w = cnf_ui.IMG_SIZE_XZ
                img_bev_b = np.full((img_h,img_w,3), 255, dtype=np.uint8)
            else:
                self.addLogs('Point cloud is empty! Try after checking radar view.')
                return
        else:
            img_bev_f, img_bev_b = get_front_beside_image_from_point_cloud(pc_os64)
        # except:
        #     self.addLogs('Bounding box is empty!')
        #     return
        
        cv2.imwrite(cnf_ui.PATH_IMG_F, img_bev_f)
        cv2.imwrite(cnf_ui.PATH_IMG_B, img_bev_b)

        # Draw bounding boxes
        plain_text = self.plainTextEditLabels.toPlainText()
        list_dict_bbox = get_list_dict_by_processing_plain_text(plain_text)

        # if len(list_dict_bbox) == 0:
        #     self.addLogs('no bboxes!')
        #     return

        # if self.spinBoxIndex_0.value() >= len(list_dict_bbox):
        #     self.addLogs('no bboxes in the index!')
        #     return

        dict_bbox = list_dict_bbox[self.spinBoxIndex_0.value()]
        
        img_front = get_front_and_beside_bev_img_with_bbox(dict_bbox, type='front')
        img_beside = get_front_and_beside_bev_img_with_bbox(dict_bbox, type='beside')

        cls_bbox = dict_bbox['cls']
        idx_cls = cnf_ui.LIST_CLS_NAME.index(cls_bbox)
        if dict_bbox['type'] == '#':    # 2D
            z_cen, z_len = cnf_ui.LIST_Z_CEN_LEN[idx_cls]
            dict_bbox['z'] = z_cen
            dict_bbox['z_l'] = z_len

            # update 2D info to 3D
            plain_text_updated = get_plain_text_with_new_dict_bbox(plain_text, \
                                                        dict_bbox, self.spinBoxIndex_0.value())
            self.plainTextEditLabels.setPlainText(plain_text_updated)

        elif dict_bbox['type'] == '*':  # 3D
            z_cen = dict_bbox['z']
            z_len = dict_bbox['z_l']

        self.label_8.setText(f'{np.round(z_len,4)} [m]')
        self.label_9.setText(f'{np.round(z_cen,4)} [m]')
        
        self.labelZf.setPixmap(get_q_pixmap_from_cv_img(img_front))
        self.labelZb.setPixmap(get_q_pixmap_from_cv_img(img_beside))

        self.is_start_z_labeling = True
    
    def pushButtonVisSensorSuit(self):
        path_pc_sensor_suit = os.path.join(cnf.BASE_DIR, 'resources', 'sensor_suit_scan.pcd')
        if os.path.exists(path_pc_sensor_suit):
            pcd = o3d.io.read_point_cloud(path_pc_sensor_suit)
            o3d.visualization.draw_geometries([pcd])
        
    def pushButtonBoundaryUp(self):
        process_z_labeling(self, type='bu')

    def pushButtonBoundaryDown(self):
        process_z_labeling(self, type='bd')

    def pushButtonCenterUp(self):
        process_z_labeling(self, type='cu')

    def pushButtonCenterDown(self):
        process_z_labeling(self, type='cd')

    ### Calibration Page ###
    def pushButtonCalibUp(self):
        calibrate_with_offset_change(self, type='u', bev_range=self.get_calib_bev_range())
    
    def pushButtonCalibLeft(self):
        calibrate_with_offset_change(self, type='l', bev_range=self.get_calib_bev_range())

    def pushButtonCalibRight(self):
        calibrate_with_offset_change(self, type='r', bev_range=self.get_calib_bev_range())

    def pushButtonCalibDown(self):
        calibrate_with_offset_change(self, type='d', bev_range=self.get_calib_bev_range())
        
    def pushButtonCalibCcw(self):
        calibrate_with_offset_change(self, type='ccw', bev_range=self.get_calib_bev_range())

    def pushButtonCalibCw(self):
        calibrate_with_offset_change(self, type='cw', bev_range=self.get_calib_bev_range())

    def pushButtonCalibBasePath(self):
        if self.path_seq_dir is None:
            path_seq = QFileDialog.getExistingDirectory(self, 'Set the folder where sequences are.', QDir.currentPath())
        else:
            path_seq = QFileDialog.getExistingDirectory(self, 'Set the folder where sequences are.', cnf_ui.PATH_SEQ)
        if not path_seq:
            self.addLogs('The folder does not exist!')
            return
        
        self.path_seq_dir = path_seq
        self.addLogs(f'Seqeunce path: {path_seq}')
        list_seq = os.listdir(path_seq)

        self.listWidgetSequence.clear()
        for idx, seq_name in enumerate(list_seq):
            temp_item = QListWidgetItem()
            temp_item.setData(1, seq_name)
            temp_item.setText(str(idx) + '. ' + seq_name)
            self.listWidgetSequence.addItem(temp_item)

    def pushButtonCalibLoad(self):
        try:
            self.addLogs(f'Seqeunce path: {self.path_seq_dir}')
            list_seq = os.listdir(self.path_seq_dir)

            self.listWidgetSequence.clear()
            for idx, seq_name in enumerate(list_seq):
                temp_item = QListWidgetItem()
                temp_item.setData(1, seq_name)
                temp_item.setText(str(idx) + '. ' + seq_name)
                self.listWidgetSequence.addItem(temp_item)
        except:
            self.addLogs('Set appropriate path!')
            return

    def pushButtonCalibGoToLabel(self):
        is_stereo = True
        if (self.dict_lidar is None) or (self.dict_radar is None):
            self.addLogs('Selecting both item is required!')
            return
        radar_idx = self.dict_radar['idx_str']
        lidar_idx = self.dict_lidar['idx_str']
        label_string = f'* radar idx: {radar_idx}, lidar idx: {lidar_idx} / Object Idx, Class, (Center [m]), Azimuth [deg], (Length [m]), Tracking Idx / 4 decimal points, # = 2D info, * = 3D info'
        self.label_23.setText(radar_idx)
        self.label_21.setText(lidar_idx)
        self.label_widget.setText(label_string)

        radar_idx = self.dict_radar['idx_str']
        lidar_idx = self.dict_lidar['idx_str']
        camera_idx = self.dict_lidar['front_img'].split(cnf_ui.SPLITTER)[-1].split('.')[0].split('_')[-1]
        time_string = self.dict_lidar['timestamp_pc']
        
        self.plainTextEditLabels.clear()
        self.plainTextEditLabels.setPlainText(f'* radar idx: {radar_idx}, lidar idx: {lidar_idx}, camera idx: {camera_idx}, time: {time_string}, prev radar idx: -1')
        
        init_bev_range = 70
        self.horizontalSliderVisRange.setValue(init_bev_range)
        self.cv_img = get_bev_img_wrt_vis_range(self, init_bev_range, self.str_time)
        self.updateBevImage(self.cv_img)

        if is_stereo:
            img_show = cv2.imread(self.dict_lidar['front_img'])[:,:1280,:]
        self.labelCam.setPixmap(get_q_pixmap_from_cv_img(cv2.resize(img_show, (cnf_ui.W_CAM, cnf_ui.H_CAM))))
        self.stackedWidget.setCurrentIndex(0)
        self.checkBox_0.setChecked(True)
        self.checkBox_7.setChecked(False)

    def itemDoubleClicked_listWidgetSequence(self):
        current_item = self.listWidgetSequence.currentItem()
        self.seq_name = current_item.data(1)
        self.addLogs(f'Current sequence = {self.seq_name}')

        path_selected_seq = os.path.join(self.path_seq_dir, self.seq_name)
        self.list_dict_lidar = get_list_dict_lidar_by_list_dir(path_selected_seq, self.seq_name)
        self.list_dict_radar = get_list_dict_radar_by_list_dir(path_selected_seq, self.seq_name)
        set_list_item_from_dict(self.listWidgetLidar, self.list_dict_lidar, data_type='lidar')
        set_list_item_from_dict(self.listWidgetRadar, self.list_dict_radar, data_type='radar')
        
    def itemDoubleClicked_listWidgetLidar(self):
        is_stereo = True
        current_item = self.listWidgetLidar.currentItem()
        self.addLogs(f'{current_item.data(0)} is selected!')
        idx_radar_frame_with_dealy = int(current_item.data(0).split('_')[1]) + self.diff_frame
        self.addLogs(f'Matched radar frame number may be = {idx_radar_frame_with_dealy}')
        self.dict_lidar = current_item.data(1)
        # print(self.dict_lidar['front_img'])
        if is_stereo:
            img_show = cv2.imread(self.dict_lidar['front_img'])[:,:1280,:]
        self.labelCalibImg.setPixmap(get_q_pixmap_from_cv_img(cv2.resize(img_show, (cnf_ui.W_CAM, cnf_ui.H_CAM))))
        
        ### is overlap checked ###
        if self.checkBox_2.isChecked():
            if self.dict_radar is None:
                self.addLogs('Selecting the radar item is required!')
                return
            else:
                if not self.checkBox_4.isChecked():
                    self.calib_offset = [0., 0., 0.]
                now_calib = (np.array(self.calib_base) + np.array(self.calib_offset)).tolist()
                get_bev_img_from_dict_radar_lidar(self.dict_radar, self.dict_lidar, bev_range=self.get_calib_bev_range(), p_frame=self, calib=now_calib)
        
        ### is overlap not checked ###
        else:
            get_bev_img_from_dict_lidar(self.dict_lidar, bev_range=self.get_calib_bev_range(), p_frame=self)

    def itemDoubleClicked_listWidgetRadar(self):
        current_item = self.listWidgetRadar.currentItem()
        self.addLogs(f'{current_item.data(0)} is selected!')
        idx_lidar_frame_with_dealy = int(current_item.data(0).split('_')[1]) - self.diff_frame
        self.addLogs(f'Matched lidar frame number may be = {idx_lidar_frame_with_dealy}')
        self.dict_radar = current_item.data(1)
        self.radar_bev_img = get_bev_img_from_dict_radar(self.dict_radar, bev_range=self.get_calib_bev_range(), p_frame=self)

    def pushButtonCalibPlayBack(self):
        msec_overlap = 20
        msec_image = 60
        len_delay = self.spinBoxDelay.value()
        if (self.list_dict_lidar is None) and (self.list_dict_radar is None):
            return
        
        len_lidar = len(self.list_dict_lidar)
        len_radar = len(self.list_dict_radar)

        if len_lidar > len_radar:
            len_actual = len_radar
        else:
            len_actual = len_lidar

        bev_range = self.get_calib_bev_range()

        H_BEV = 480
        W_BEV = 768

        is_reverse = False
        if len_delay < 0:
            is_reverse = True
            len_delay = -len_delay

        for i in range(len_actual):
            try:
                if is_reverse:
                    temp_dict_lidar = self.list_dict_lidar[len_delay:][i]
                    temp_dict_radar = self.list_dict_radar[i]
                else:
                    temp_dict_lidar = self.list_dict_lidar[i]
                    temp_dict_radar = self.list_dict_radar[len_delay:][i]


                img_front = cv2.imread(temp_dict_lidar['front_img'])
                # img_front = img_front[:,1280:,:] # R
                img_front = img_front[:,:1280,:] # L
                img_front = cv2.resize(img_front, (854, 480), interpolation=cv2.INTER_LINEAR)
                
                if self.checkBox_3.isChecked():
                    now_calib = np.array(self.calib_base)+np.array(self.calib_offset)
                    now_calib = now_calib.tolist()
                    img_overlap = get_bev_img_from_dict_radar_lidar(temp_dict_radar, temp_dict_lidar, bev_range, self, now_calib, \
                                                                                        is_visualize=False, is_update_str=False)

                    if self.checkBox_6.isChecked():
                        temp_key = f'bev_{bev_range}'
                        img_lidar = cv2.imread(temp_dict_lidar[temp_key])
                        img_radar = cv2.imread(temp_dict_radar[temp_key])
                        cv2.imwrite(f'./saved_frames/camera_{i}.png', img_front)
                        cv2.imwrite(f'./saved_frames/lidar_{i}.png', img_lidar)
                        cv2.imwrite(f'./saved_frames/radar_{i}.png', img_radar)
                        cv2.imwrite(f'./saved_frames/overlap_{i}.png', img_overlap)
                        
                    img_overlap = cv2.resize(img_overlap, (W_BEV, H_BEV), interpolation=cv2.INTER_LINEAR)
                    img_total = cv2.hconcat([img_front, img_overlap])

                    cv2.imshow('RA4D', img_total)
                    waitKey = cv2.waitKey(msec_overlap)
                else:
                    temp_key = f'bev_{bev_range}'
                    img_lidar = cv2.imread(temp_dict_lidar[temp_key])
                    img_lidar = cv2.resize(img_lidar, (W_BEV, H_BEV), interpolation=cv2.INTER_LINEAR)

                    img_radar = cv2.imread(temp_dict_radar[temp_key])
                    img_radar = cv2.resize(img_radar, (W_BEV, H_BEV), interpolation=cv2.INTER_LINEAR)

                    img_total = cv2.hconcat([img_front, img_lidar, img_radar])
                    cv2.imshow('RA4D', img_total)
                    waitKey = cv2.waitKey(msec_image)

                if waitKey == 113:
                    print(temp_dict_lidar[temp_key])
                    print(temp_dict_radar[temp_key])
                    cv2.destroyWindow('RA4D')
                    return
            except:
                return
    
    def pushButtonUpdateBaseParams(self):
        self.calib_base = (np.array(self.calib_base) + np.array(self.calib_offset)).tolist()
        self.calib_offset = [0., 0., 0.]

        x_b, y_b, yaw_b = self.calib_base
        x_o, y_o, yaw_o = self.calib_offset
        self.label_27.setText('%+.3f %+.3f [m]' % (x_b, x_o))
        self.label_28.setText('%+.3f %+.3f [m]' % (y_b, y_o))
        self.label_31.setText('%+.3f %+.3f [deg]' % (yaw_b, yaw_o))

    def get_calib_bev_range(self):
        list_bev_range = ['15', '30', '50', '100']
        for i in range(4):
            if getattr(self, f'radioButtonCalib_{i}').isChecked():
                return list_bev_range[i]
        assert True, 'Check radio button!'
    
    def pushButtonCalibUpdateBev(self):
        if (self.dict_lidar is None) or (self.dict_radar is None):
            self.addLogs('Selecting both item is required!')
            return
        
        now_calib = (np.array(self.calib_base) + np.array(self.calib_offset)).tolist()
        get_bev_img_from_dict_radar_lidar(self.dict_radar, self.dict_lidar, bev_range=self.get_calib_bev_range(), p_frame=self, calib=now_calib)

    def pushButtonCalibSaveFixed(self):
        if len(self.plainTextEditLabels.toPlainText()) < 5:
            self.addLogs('Generate the matching info first!')
            return

        idx_radar = self.dict_radar['idx_str']
        idx_lidar = self.dict_lidar['idx_str']
        
        path_selected_seq = os.path.join(self.path_seq_dir, self.seq_name)
        file_name = f'{idx_radar}_{idx_lidar}.txt'
        f = open(os.path.join(path_selected_seq, 'info_matching', file_name), 'w')
        f.write(self.plainTextEditLabels.toPlainText())
        f.close()

        self.addLogs(f'{file_name} is saved to {path_selected_seq}/info_matching')

    def pushButtonCalibSaveDialog(self):
        if len(self.plainTextEditLabels.toPlainText()) < 5:
            self.addLogs('Generate the matching info first!')
            return

    def pushButtonCalibGenerateText(self):
        if (self.dict_lidar is None) or (self.dict_radar is None):
            self.addLogs('Selecting both item is required!')
            return
        
        splitter = cnf_ui.SPLITTER

        calib_base = self.calib_base
        calib_offset = self.calib_offset
        calib_total = (np.array(calib_base) + np.array(calib_offset)).tolist()

        file_radar = (self.dict_radar['tesseract']).split(splitter)[-1]
        file_lidar = (self.dict_lidar['pc']).split(splitter)[-1]
        file_camera = (self.dict_lidar['front_img']).split(splitter)[-1]

        plain_text_info_matching = ''
        plain_text_info_matching = plain_text_info_matching + f'{file_radar}, {file_lidar}, {file_camera}\n'
        plain_text_info_matching = plain_text_info_matching + '%+.3f, %+.3f, %+.3f\n' % (calib_total[0], calib_total[1], calib_total[2])
        plain_text_info_matching = plain_text_info_matching + '%+.3f, %+.3f, %+.3f\n' % (calib_base[0], calib_base[1], calib_base[2])
        plain_text_info_matching = plain_text_info_matching + '%+.3f, %+.3f, %+.3f' % (calib_offset[0], calib_offset[1], calib_offset[2])

        self.plainTextEditLabels.clear()
        self.plainTextEditLabels.setPlainText(plain_text_info_matching)

    def pushButtonSaveLabel(self):
        if len(self.plainTextEditLabels.toPlainText()) < 5:
            self.addLogs('Generate the label info first!')
            return

        idx_radar = self.dict_radar['idx_str']
        idx_lidar = self.dict_lidar['idx_str']
        
        path_selected_seq = os.path.join(self.path_seq_dir, self.seq_name)
        file_name = f'{idx_radar}_{idx_lidar}.txt'
        f = open(os.path.join(path_selected_seq, 'info_label', file_name), 'w')
        f.write(self.plainTextEditLabels.toPlainText())
        f.close()

        self.addLogs(f'{file_name} is saved to {path_selected_seq}/info_label')

    def pushButtonLoadLabel(self):
        path_selected_seq = os.path.join(self.path_seq_dir, self.seq_name)
        path_file = QFileDialog.getOpenFileName(self, 'Select the label info to get', os.path.join(path_selected_seq, 'info_label'))
        
        if path_file[0] == '':
            return
        
        f = open(path_file[0], 'r')
        lines = f.readlines()
        str_info = ''
        for line in lines:
            str_info += line
        f.close()

        self.plainTextEditLabels.setPlainText(str_info)

    def pushButtonLoadLabelPrev(self):
        path_selected_seq = os.path.join(self.path_seq_dir, self.seq_name)
        path_file = QFileDialog.getOpenFileName(self, 'Select the previous label info to get', os.path.join(path_selected_seq, 'info_label'))
        
        if path_file[0] == '':
            return
        
        # self.dict_lidar['idx_prev_str'] = 

        f = open(path_file[0], 'r')
        lines = f.readlines()
        str_info = ''
        for idx_line, line in enumerate(lines):
            if idx_line == 0:
                str_info += line
            else:
                # replace prev idx with obj idx
                list_comp = line.split(',')
                list_comp[2] = list_comp[1]
                list_comp[1] = f' {idx_line-1}'
                str_info += ','.join(map(str, list_comp))
        f.close()

        self.plainTextEditLabels.setPlainText(str_info)

    def pushButtonRotateCcw(self):
        updateModifiedBboxInfo(self, 'ccw', self.doubleSpinBoxHeading.value() ,self.spinBoxIndex_0.value())

    def pushButtonRotateCw(self):
        updateModifiedBboxInfo(self, 'cw', self.doubleSpinBoxHeading.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxUp(self):
        updateModifiedBboxInfo(self, 'u', self.doubleSpinBoxTranslation.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxDown(self):
        updateModifiedBboxInfo(self, 'd', self.doubleSpinBoxTranslation.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxLeft(self):
        updateModifiedBboxInfo(self, 'l', self.doubleSpinBoxTranslation.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxRight(self):
        updateModifiedBboxInfo(self, 'r', self.doubleSpinBoxTranslation.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxLxUp(self):
        updateModifiedBboxInfo(self, 'xu', self.doubleSpinBoxSize.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxLxDown(self):
        updateModifiedBboxInfo(self, 'xd', self.doubleSpinBoxSize.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxLyDown(self):
        updateModifiedBboxInfo(self, 'yd', self.doubleSpinBoxSize.value() ,self.spinBoxIndex_0.value())

    def pushButtonModifyBoxLyUp(self):
        updateModifiedBboxInfo(self, 'yu', self.doubleSpinBoxSize.value() ,self.spinBoxIndex_0.value())

    def pushButtonBackToCalib(self):
        calib_string = '* matching info: radar, lidar, camera / total_calib: x_t [m], y_t [m], yaw_t [deg] / base_calib: x_b [m], y_b [m], yaw_b [deg] / offset_calib: x_o [m], y_o [m], yaw_o [deg]'
        self.plainTextEditLabels.clear()
        self.plainTextEditLabels.setPlainText(calib_string)

        path_selected_seq = os.path.join(self.path_seq_dir, self.seq_name)
        self.list_dict_lidar = get_list_dict_lidar_by_list_dir(path_selected_seq, self.seq_name)
        self.list_dict_radar = get_list_dict_radar_by_list_dir(path_selected_seq, self.seq_name)
        set_list_item_from_dict(self.listWidgetLidar, self.list_dict_lidar, data_type='lidar')
        set_list_item_from_dict(self.listWidgetRadar, self.list_dict_radar, data_type='radar')
        
        if not (self.name_labeler is None):
            self.textEditNameLabeler.setText(self.name_labeler)
        self.checkBox_0.setChecked(False)
        self.stackedWidget.setCurrentIndex(1)

    def pushButtonSetFontSize(self):
        updateUiDetails(self, self.spinBoxFont.value())
        
    def pushButtonSetFrameDiff(self):
        self.diff_frame = self.spinBoxDelay.value()
        self.addLogs(f'Frame delay is set as {self.diff_frame}.')

    def pushButtonFrontImg(self):
        type_stereo = 'left' if self.horizontalSliderStereo.value() == 0 else 'right'
        showImageFourDirections(self, type_cam='front', stereo=type_stereo)

    def pushButtonLefImg(self):
        type_stereo = 'left' if self.horizontalSliderStereo.value() == 0 else 'right'
        showImageFourDirections(self, type_cam='left', stereo=type_stereo)

    def pushButtonRightImg(self):
        type_stereo = 'left' if self.horizontalSliderStereo.value() == 0 else 'right'
        showImageFourDirections(self, type_cam='right', stereo=type_stereo)

    def pushButtonRearImg(self):
        type_stereo = 'left' if self.horizontalSliderStereo.value() == 0 else 'right'
        showImageFourDirections(self, type_cam='rear', stereo=type_stereo)

    def pushButtonVisualizeBBox(self):
        path_pcd = self.dict_lidar['pc']
        pcd = o3d.io.read_point_cloud(path_pcd)
        list_pcd = []
        list_pcd.append(pcd)

        plain_text = self.plainTextEditLabels.toPlainText()
        list_dict_bbox = get_list_dict_by_processing_plain_text(plain_text)

        for idx, dict_bbox in enumerate(list_dict_bbox):
            if dict_bbox['type'] == '#':
                list_infos = [dict_bbox['x'], dict_bbox['y'], \
                    dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l']]
                self.addLogs('Error: There is a 2D bbox!')
                continue
            elif dict_bbox['type'] == '*':
                list_infos = [dict_bbox['x'], dict_bbox['y'], dict_bbox['z'], \
                    dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l'], dict_bbox['z_l']]
                _, color_bgr = self.getClsNameAndColor(dict_bbox['cls'])
                color_rgb_norm = [color_bgr[2]/255., color_bgr[1]/255., color_bgr[0]/255.]
                list_pcd.append(get_o3d_line_set_from_list_infos(list_infos, color=color_rgb_norm))

        o3d.visualization.draw_geometries(list_pcd)

    def pushButtonNameLabeler(self):
        self.name_labeler = self.textEditNameLabeler.toPlainText()
        self.addLogs(f'Labeler: {self.name_labeler}')
        self.addLogs(f'Starting at: {get_now_time_string()}')

        if (self.path_seq_dir is None) or (self.seq_name is None):
            return

        if self.idx_file_log is None:
            self.idx_file_log = len(os.listdir(os.path.join(self.path_seq_dir, self.seq_name, 'info_frames')))
        f = open(os.path.join(self.path_seq_dir, self.seq_name, 'info_frames', f'log_{self.idx_file_log}.txt'), 'w')
        now_time = get_now_time_string()
        txt_log = f'labeler = {self.name_labeler}\n' + f'time = {now_time}\n'
        f.write(txt_log)
        f.close()

    def pushButtonShowTrackInfo(self):
        print('hi')
        
    def pushButtonLcChangeSequence(self):
        self.pushButtonGoToLcCalib()

    def pushButtonGoToLcCalib(self):
        dir_header = QFileDialog.getExistingDirectory(self, 'Select Directory ', '')
        
        self.textEditLcCalib_3.setText(dir_header)
        self.timestamps = [] # time stamp
        self.items = [] # cam, ldr, rdr, ...
        
        self.info_labels = self.get_info_label(f"{dir_header}/info_label")
        
        for timestamps_abs_file in self.info_labels:
            items, timestamp = self.get_timestamp_and_indices(timestamps_abs_file) 
            self.timestamps.append(timestamp)
            self.items.append(items) 

        print(len(self.timestamps))
        self.spinBoxLcCalib_0.setMaximum(len(self.timestamps)-1)   

        ### Init required values ###
        self.list_lc_calib_keys = [
            'fx', 'fy', 'px', 'py', \
            'k1', 'k2', 'k3', 'k4', 'k5', \
            'roll_ldr2cam', 'pitch_ldr2cam', 'yaw_ldr2cam', 'x_ldr2cam', 'y_ldr2cam', 'z_ldr2cam'
        ]
        list_lc_calib_init_values = [
            567.720776478944, 577.2136917114258, 628.720776478944, 369.3068656921387, \
            -0.028873818023371287, 0.0006023302214797655, 0.0039573086622276855, -0.005047176298643093, 0.0, \
             0.0, 1.9, 0.0, 0.1, 0.0, -0.55
        ]

        for i in range(15):
            getattr(self, f'doubleSpinBoxLcCalib_{i}').setValue(list_lc_calib_init_values[i])

        self.dict_lc_calib_init_values = dict()
        self.dict_lc_calib_now_values = dict()
        for idx, param in enumerate(self.list_lc_calib_keys):
            self.dict_lc_calib_init_values.update(
                {param:list_lc_calib_init_values[idx]}
            )
            self.dict_lc_calib_now_values.update(
                {param:list_lc_calib_init_values[idx]}
            )

        for i in range(15):
            getattr(self, f'doubleSpinBoxLcCalib_{i}').valueChanged.\
                connect(getattr(self, f'doubleSpinBoxLcCalibValueChanged_{i}'))
                
        self.ChangeData("Front(L)")

        self.stackedWidget.setCurrentIndex(2)

    def doubleSpinBoxLcCalibValueChanged(self, i, value):
        key = self.list_lc_calib_keys[i]
        self.dict_lc_calib_now_values[key] = value


    def doubleSpinBoxLcCalibValueChanged_0(self): self.doubleSpinBoxLcCalibValueChanged(0, self.doubleSpinBoxLcCalib_0.value())
    def doubleSpinBoxLcCalibValueChanged_1(self): self.doubleSpinBoxLcCalibValueChanged(1, self.doubleSpinBoxLcCalib_1.value())
    def doubleSpinBoxLcCalibValueChanged_2(self): self.doubleSpinBoxLcCalibValueChanged(2, self.doubleSpinBoxLcCalib_2.value())
    def doubleSpinBoxLcCalibValueChanged_3(self): self.doubleSpinBoxLcCalibValueChanged(3, self.doubleSpinBoxLcCalib_3.value())
    def doubleSpinBoxLcCalibValueChanged_4(self): self.doubleSpinBoxLcCalibValueChanged(4, self.doubleSpinBoxLcCalib_4.value())
    def doubleSpinBoxLcCalibValueChanged_5(self): self.doubleSpinBoxLcCalibValueChanged(5, self.doubleSpinBoxLcCalib_5.value())
    def doubleSpinBoxLcCalibValueChanged_6(self): self.doubleSpinBoxLcCalibValueChanged(6, self.doubleSpinBoxLcCalib_6.value())
    def doubleSpinBoxLcCalibValueChanged_7(self): self.doubleSpinBoxLcCalibValueChanged(7, self.doubleSpinBoxLcCalib_7.value())
    def doubleSpinBoxLcCalibValueChanged_8(self): self.doubleSpinBoxLcCalibValueChanged(8, self.doubleSpinBoxLcCalib_8.value())

    def doubleSpinBoxLcCalibValueChanged_9(self): self.doubleSpinBoxLcCalibValueChanged(9, self.doubleSpinBoxLcCalib_9.value())
    def doubleSpinBoxLcCalibValueChanged_10(self): self.doubleSpinBoxLcCalibValueChanged(10, self.doubleSpinBoxLcCalib_10.value())
    def doubleSpinBoxLcCalibValueChanged_11(self): self.doubleSpinBoxLcCalibValueChanged(11, self.doubleSpinBoxLcCalib_11.value())
    def doubleSpinBoxLcCalibValueChanged_12(self): self.doubleSpinBoxLcCalibValueChanged(12, self.doubleSpinBoxLcCalib_12.value())
    def doubleSpinBoxLcCalibValueChanged_13(self): self.doubleSpinBoxLcCalibValueChanged(13, self.doubleSpinBoxLcCalib_13.value())
    def doubleSpinBoxLcCalibValueChanged_14(self): self.doubleSpinBoxLcCalibValueChanged(14, self.doubleSpinBoxLcCalib_14.value())
    # def doubleSpinBoxLcCalibValueChanged_15(self): self.doubleSpinBoxLcCalibValueChanged(15, self.doubleSpinBoxLcCalib_15.value())
    # def doubleSpinBoxLcCalibValueChanged_16(self): self.doubleSpinBoxLcCalibValueChanged(16, self.doubleSpinBoxLcCalib_16.value())
    # def doubleSpinBoxLcCalibValueChanged_17(self): self.doubleSpinBoxLcCalibValueChanged(17, self.doubleSpinBoxLcCalib_17.value())

    def pushButtonLcBackToRlCalib(self):
        self.stackedWidget.setCurrentIndex(1)
        
    def pushButtonLcShowSensorSuite(self):
        print('hi')

    def pushButtonLcInitValue(self):
        list_lc_calib_init_values = [
            567.720776478944, 577.2136917114258, 628.720776478944, 369.3068656921387, \
            -0.028873818023371287, 0.0006023302214797655, 0.0039573086622276855, -0.005047176298643093, 0.0, \
            0.0, 1.9, 0.0, 0.1, 0.0, -0.55
        ]
        
        for i in range(15):
            getattr(self, f'doubleSpinBoxLcCalib_{i}').setValue(list_lc_calib_init_values[i])    
        # new_dict_calib_values, new_dict_calib_offsets = \
        #     get_dict_lc_calib_from_txt(self.textEditLcCalib_0.toPlainText(), self.list_lc_calib_keys)
        # self.dict_lc_calib_init_values.update(new_dict_calib_values)
        # self.dict_lc_calib_now_values.update(new_dict_calib_values)
        # self.dict_lc_calib_offsets.update(new_dict_calib_offsets)
        
        # set_txt_label_dict_lc_calib(self, self.list_lc_calib_keys, self.dict_lc_calib_now_values)

    def pushButtonLcShowCalib(self):
        img_size = (1280, 720)
        # tr_rotation_default = np.array([
        #     [0.0, 0.0, -1.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0],
        #     [1.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 1.0]
        # ])
        tr_rotation_default = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        intrinsics, distortion, r_cam, tr_lid_cam =  get_matrices_from_dict_lc_calib(self.dict_lc_calib_now_values)

        print("Transformation(ldr2cam)")
        print(tr_lid_cam)

        img_process = self.img_lc

        if self.checkBox_distortion.isChecked():
            ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
            for j in range(3):
                for i in range(3):
                    intrinsics[j,i] = ncm[j, i]
            
            map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, r_cam, ncm, img_size, cv2.CV_32FC1)
            img_process = cv2.remap(img_process, map_x, map_y, cv2.INTER_LINEAR)

        ### Get additional attribute for point cloud ###
        pcd = o3d.io.read_point_cloud(self.path_pcd_lc)
        list_attributes = ['intensity', 'reflectivity', 'ring']
        dict_values = dict()
        for attribute in list_attributes:
            dict_values.update({attribute: read_attribute_from_pcd(attribute, self.path_pcd_lc)})
        ### Get additional attribute for point cloud ###

        points_roi = np.array(np.asarray(pcd.points))
        if self.checkBox_pc_roi.isChecked():
            x_min, x_max, y_min, y_max, z_min, z_max = get_pc_roi_from_txt(self.textEditLcCalib_1.toPlainText())
            
            points = np.array(np.asarray(pcd.points))
            list_for_concatenation = [points]
            for attribute in list_attributes:
                list_for_concatenation.append(dict_values[attribute])
            points_roi = np.concatenate(list_for_concatenation, axis=1)
            points_roi = np.array(list(filter(lambda x: (x[0]>x_min) and (x[0]<x_max) \
                and (x[1]>y_min) and (x[1]<y_max) and (x[2]>z_min) and (x[2]<z_max), points_roi.tolist())))
            pcd.points = o3d.utility.Vector3dVector(points_roi[:,:3])
            values_roi = points_roi[:,3:]
            for idx_temp, attribute in enumerate(list_attributes):
                dict_values.update({attribute: values_roi[:,idx_temp:idx_temp+1]})
            for idx_temp, attribute in enumerate(['x', 'y', 'z']):
                dict_values.update({attribute: points_roi[:,idx_temp:idx_temp+1]})
        
        P2 = np.insert(intrinsics, 3, values=[0,0,0], axis=1) # K = intrinsics
        R0_hom = r_cam
        R0_hom = np.insert(R0_hom,3,values=[0,0,0],axis=0)
        R0_hom = np.insert(R0_hom,3,values=[0,0,0,1],axis=1) # 4x4
        points_hom = np.transpose(points_roi[:,:3].copy(), (1,0))
        points_hom = np.insert(points_hom, 3, 1, axis=0) # 4xN
        LidarToCamera = np.insert(tr_lid_cam, 3, values=[0,0,0,1], axis=0)

        # Adding default rotation
        cam = np.matmul(np.matmul(np.matmul(P2,R0_hom),np.matmul(LidarToCamera,tr_rotation_default)),points_hom)
        # cam = np.matmul(np.matmul(np.matmul(P2,R0_hom),LidarToCamera),points_hom)
        cam[:2] /= cam[2,:]
        png = np.flip(img_process, axis=2) # bgr to rgb

        plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
        IMG_H,IMG_W,_ = png.shape
        plt.axis([0,IMG_W,IMG_H,0])
        plt.imshow(png)
        u,v,z = cam
        u_out = np.logical_or(u<0, u>IMG_W)
        v_out = np.logical_or(v<0, v>IMG_H)
        outlier = np.logical_or(u_out, v_out)
        cam = np.delete(cam,np.where(outlier),axis=1)
        u,v,z = cam
        plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.2,s=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./resources/imgs/frame/temp_img_result.png')
        plt.savefig('./resources/imgs/frame/temp_img.png', bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()

        self.labelLidCamCalibImg.setPixmap(get_q_pixmap_from_cv_img(cv2.resize(cv2.imread('./resources/imgs/frame/temp_img.png'), (1280,720), cv2.INTER_LINEAR)))

    def pushButtonLcShowRoiPc(self):
        pcd = o3d.io.read_point_cloud(self.path_pcd_lc)
        list_attributes = ['intensity', 'reflectivity', 'ring']
        dict_values = dict()
        for attribute in list_attributes:
            dict_values.update({attribute: read_attribute_from_pcd(attribute, self.path_pcd_lc)})
        points_roi = np.array(np.asarray(pcd.points))

        if self.checkBox_pc_roi.isChecked():
            x_min, x_max, y_min, y_max, z_min, z_max = get_pc_roi_from_txt(self.textEditLcCalib_1.toPlainText())

            points = np.array(np.asarray(pcd.points))
            list_for_concatenation = [points]   
            for attribute in list_attributes:
                list_for_concatenation.append(dict_values[attribute])
            points_roi = np.concatenate(list_for_concatenation, axis=1)
            points_roi = np.array(list(filter(lambda x: (x[0]>x_min) and (x[0]<x_max) \
                and (x[1]>y_min) and (x[1]<y_max) and (x[2]>z_min) and (x[2]<z_max), points_roi.tolist())))
            pcd.points = o3d.utility.Vector3dVector(points_roi[:,:3])
            values_roi = points_roi[:,3:]
            for idx_temp, attribute in enumerate(list_attributes):
                dict_values.update({attribute: values_roi[:,idx_temp:idx_temp+1]})
            for idx_temp, attribute in enumerate(['x', 'y', 'z']):
                dict_values.update({attribute: points_roi[:,idx_temp:idx_temp+1]})
        
        ### Colorization ###
        hsv_criterion = dict_values['reflectivity'] # keys: x, y, z, intensity, reflectivity, ring
        rgb_values = get_hsv_to_rgb_via_min_max_values(hsv_criterion, sat=1.0, val=1.0, normalize_method='mix_1')
        ### Colorization ###

        ### Visualize point cloud ###
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_roi[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(rgb_values)
        # pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(points_roi[:,:3]))
        o3d.visualization.draw_geometries([pcd])
        ### Visualize point cloud ###
        
    def pushButtonLcFrontLeft(self): self.ChangeData("Front(L)")
    def pushButtonLcFrontRight(self): self.ChangeData("Front(R)")
    def pushButtonLcRearLeft(self): self.ChangeData("Rear(L)")
    def pushButtonLcRearRight(self): self.ChangeData("Rear(R)")
    def pushButtonLcLeftLeft(self): self.ChangeData("Left(L)")
    def pushButtonLcLeftRight(self): self.ChangeData("Left(R)")
    def pushButtonLcRightLeft(self): self.ChangeData("Right(L)")
    def pushButtonLcRightRight(self): self.ChangeData("Right(R)")
    
    def ChangeData(self, cam_name:str):
        self.textEditLcCalib_2.setText(cam_name)
        
        cam_direction = f"cam-{cam_name.split('(')[0].lower()}"
        stereo_LR = cam_name[-2].lower()
        self.cam_direction = cam_direction
        self.stereo_LR = stereo_LR
        idx = int(self.spinBoxLcCalib_0.value())        
            
        self.switchROI(self.cam_direction.split('-')[-1])

        self.setCondition(cam_direction, stereo_LR, idx)

    def idx_changed(self):
        idx = int(self.spinBoxLcCalib_0.value())
        self.setCondition(self.cam_direction, self.stereo_LR, idx)


    def switchROI(self, cam_direction):
        # x_min, x_max, y_min, y_max, z_min, z_max = get_pc_roi_from_txt(self.textEditLcCalib_1.toPlainText())
        result = dict()
        roi_info = self.textEditLcCalib_1.toPlainText().split('\n')[1:]
        if roi_info[-1] == '':
            roi_info = roi_info[:-1]
        
        keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        for info in roi_info:
            result[info.split(':')[0]] = float(info.split(':')[1])
        
        if cam_direction == 'front':
            result['x_min'] = 0.
            result['x_max'] = 100.
            result['y_min'] = -100.
            result['y_max'] = 100.
        elif cam_direction == 'rear':
            result['x_min'] = -100.
            result['x_max'] = 0.
            result['y_min'] = -100.
            result['y_max'] = 100.
        elif cam_direction == 'left':
            result['x_min'] = -100.
            result['x_max'] = 100.
            result['y_min'] = 0.
            result['y_max'] = 100.
        elif cam_direction == 'right':
            result['x_min'] = -100.
            result['x_max'] = 100.
            result['y_min'] = -100.
            result['y_max'] = 0.
        
               
        text = "[m]\n"
        for key in keys:
            text += f"{key}: {result[key]}\n"
        
        self.textEditLcCalib_1.setText(text)

    def setCondition(self, cam_direction, stereo_LR, img_idx=0):
            
        path_header = self.textEditLcCalib_3.toPlainText()
        cam_timestamp = cam_direction if cam_direction == "cam-front" else "cam-lrr"
        
        
        self.path_pcd_lc = f"{path_header}/os2-64/os2-64_{str(self.items[img_idx]['os2-64']).zfill(5)}.pcd"
        self.path_label_lc = f"{path_header}/info_label/{str(self.items[img_idx]['info_label'])}.txt"
        path_img = f"{path_header}/{cam_direction}/{cam_direction}_{str(self.items[img_idx][cam_timestamp]).zfill(5)}.png"
        
        # 33,1 / 629, 597
        self.img_lc = cv2.imread(path_img)
        if stereo_LR == 'l': # left
            self.img_lc = self.img_lc[:,:1280,:].copy() # left
        else:
            self.img_lc = self.img_lc[:,1280:,:].copy() #right
        self.labelLidCamCalibImg.setPixmap(get_q_pixmap_from_cv_img(self.img_lc))


        for i in range(15):
            getattr(self, f'doubleSpinBoxLcCalib_{i}').valueChanged.\
                connect(getattr(self, f'doubleSpinBoxLcCalibValueChanged_{i}'))
                
        
    def pushButtonLcSaveParams(self):
        filepath=savepath
        cam_number = self.assign_cam_number(self.cam_direction, self.stereo_LR)
        params = dict()
        params.update({'cam_number':cam_number}) 
        params.update(self.dict_lc_calib_now_values)
        
        self.save_camera_params(f"{filepath}/cam_{cam_number}.yml", params)
        
    def pushButtonLcLoadParams(self):
        try:
            cam_number = self.assign_cam_number(self.cam_direction, self.stereo_LR)
            filepath = loadpath
            params = self.load_camera_params(f"{filepath}/cam_{cam_number}.yml")        
            _ = params.pop("cam_number", None)
            self.dict_lc_calib_now_values = params.copy()            
            list_lc_calib_init_values = [self.dict_lc_calib_now_values[self.list_lc_calib_keys[i]] for i in range(15)]
        except:
            print("you should save the camera params at first.")       
            list_lc_calib_init_values = [
            567.720776478944, 577.2136917114258, 628.720776478944, 369.3068656921387, \
            -0.028873818023371287, 0.0006023302214797655, 0.0039573086622276855, -0.005047176298643093, 0.0, \
            0.0, 1.9, 0.0, 0.1, 0.0, -0.55
            ]
        for i in range(15):
            getattr(self, f'doubleSpinBoxLcCalib_{i}').setValue(list_lc_calib_init_values[i])
    
    def assign_cam_number(self, cam_direction, stereo_LR):
        cam_direction = cam_direction.split('-')[-1]

        if cam_direction == 'front':
            cam_number = 1 if stereo_LR == 'l' else 2
        elif cam_direction == 'right':
            cam_number = 3 if stereo_LR == 'l' else 4
        elif cam_direction == 'rear':
            cam_number = 5 if stereo_LR == 'l' else 6
        elif cam_direction == 'left':
            cam_number = 7 if stereo_LR == 'l' else 8
        
        return cam_number
        
        
    def save_camera_params(self, yaml_file_path, settings):
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(settings, yaml_file, default_flow_style=False)
        print(f'{yaml_file_path.split("/")[-1]} has been saved in {yaml_file_path}')

    def load_camera_params(self, yaml_file_path):
        if os.path.exists(yaml_file_path):
            with open(yaml_file_path, 'r') as yaml_file:
                settings = yaml.safe_load(yaml_file)
                return settings
        else:
            print(f'{yaml_file_path} does not exists.')
            return None
        
        
    def get_info_label(self, dir_path):
        try:
            files = os.listdir(dir_path)
            files = sorted(files, key = lambda x: int(x.split("_")[-1].split(".")[0]))
            files = [f"{dir_path}/{filename}" for filename in files]
        except FileNotFoundError:
            print(f"'{dir_path}' doesn't exist.")
        return files
    
    def get_timestamp_and_indices(self, txt_file_path):
        items, timestamp = None, None
        try:
            with open(txt_file_path, 'r') as file:
                items, timestamp= file.readline().split(',')
                keys, values = items.split('=') # name, value
                # get tesseract, os2-64, cam-front, os1-128, and cam-lrr 
                keys = keys.split('_')
                keys[0] = keys[0].split('(')[-1]
                keys[-1] = keys[-1].split(')')[0]
                # match each values
                values = [int(value) for value in values.split('_')]
                items = dict(zip(keys, values))                
                timestamp = float(timestamp.split('=')[-1])
        except FileNotFoundError:
            print(f"'{txt_file_path}' doesn't exist.")
        except Exception as e:
            print(f"Error happens for reading file: {str(e)}")            
        items['info_label'] = txt_file_path.split('/')[-1]
        return items, timestamp

def startUi():
    app = QApplication(sys.argv)
    main_frame = MainFrame()
    main_frame.show()
    sys.exit(app.exec_())
