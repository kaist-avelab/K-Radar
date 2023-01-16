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
import numpy as np
import cv2

from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

__all__ = [ 'get_q_pixmap_from_cv_img', \
            'get_intrinsic_and_extrinsic_params_from_text_edit', \
            'get_list_p_text_edit', \
            'get_rotation_and_translation_from_extrinsic', \
            'get_pixel_from_point_cloud_in_camera_coordinate', \
            'get_pointcloud_with_rotation_and_translation', ]

def get_q_pixmap_from_cv_img(cv_img, width=None, height=None, interpolation=cv2.INTER_LINEAR):
    if width and height:
        cv_img = cv2.resize(cv_img, dsize=(width, height), interpolation=interpolation)
    
    return QPixmap.fromImage(cv_img_to_q_image(cv_img))

def cv_img_to_q_image(cv_img):
    if len(np.shape(cv_img))==2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

    height, width, _ = cv_img.shape

    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    q_img = q_img.rgbSwapped()

    return q_img

def get_intrinsic_and_extrinsic_params_from_text_edit(list_p_text_edit):
    '''
    * return
    *   1. intrinsic (fx, fy, px, py)
    *   2. extrinsic (roll, pitch, yaw, x, y, z) [degree, m]
    '''
    assert len(list_p_text_edit) == 4, 'get 4 rows'

    list_rows = list(map(lambda text_edit: text_edit.toPlainText(), list_p_text_edit))
    list_list_params = []
    temp_list_params = list(map(lambda plain_text: plain_text.split(','), list_rows))
    for list_params in temp_list_params:
        list_params = list(map(lambda param: float(param), list_params))
        list_list_params.append(list_params)
    
    intrinsic = [list_list_params[0][0], list_list_params[1][1], list_list_params[0][2], list_list_params[1][2]]
    extrinsic = []
    extrinsic.extend(list_list_params[2])
    extrinsic.extend(list_list_params[3])

    return intrinsic, extrinsic

def get_list_p_text_edit(p_mf):
    list_p_text_edit = []
    
    for i in range(4):
        list_p_text_edit.append(getattr(p_mf, 'plainTextEdit_cal_'+str(i+1)))
    
    return list_p_text_edit

def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg = True):
    ext_copy = extrinsic.copy() # if not copy, will change the parameters permanently
    if is_deg:
        ext_copy[:3] = list(map(lambda x: x*np.pi/180., extrinsic[:3]))

    roll, pitch, yaw = ext_copy[:3]
    x, y, z = ext_copy[3:]

    ### Roll-Pitch-Yaw Convention
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    R_yaw = np.array([[c_y, -s_y, 0.],[s_y, c_y, 0.],[0., 0., 1.]])
    R_pitch = np.array([[c_p, 0., s_p],[0., 1., 0.],[-s_p, 0., c_p]])
    R_roll = np.array([[1., 0., 0.],[0., c_r, -s_r],[0., s_r, c_r]])

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
    trans = np.array([[x],[y],[z]])

    return R, trans

def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
    '''
    * in : pointcloud in np array (nx3)
    * out: projected pixel in np array (nx2)
    '''

    process_pc = point_cloud_xyz.copy()
    if (np.shape(point_cloud_xyz) == 1):
        num_points = 0
    else:
        #Temporary fix for when shape = (0.)
        try:
            num_points, _ = np.shape(point_cloud_xyz)
        except:
            num_points = 0
    fx, fy, px, py = intrinsic

    pixels = []
    for i in range(num_points):
        xc, yc, zc = process_pc[i,:]
        y_pix = py - fy*zc/xc
        x_pix = px - fx*yc/xc

        pixels.append([x_pix, y_pix])
    pixels = np.array(pixels)

    return pixels

def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
    pc_xyz = point_cloud_xyz.copy()
    num_points = len(pc_xyz)
    
    for i in range(num_points):
        point_temp = pc_xyz[i,:]
        point_temp = np.reshape(point_temp, (3,1))

        point_processed = np.dot(rot, point_temp) + tra
        point_processed = np.reshape(point_processed, (3,))
        
        pc_xyz[i,:] = point_processed

    return pc_xyz
