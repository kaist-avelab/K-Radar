"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.10.06
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for ui configs
"""

# Library
import os.path as osp

# User Library
from configs.config_general import BASE_DIR, IS_UBUNTU

# Calibration
# Default Calibration Values: Translation -> Rotation / Lidar -> Radar: Reference is Radar
CALIB = [-2.54, 0.3, 0.] # [m, m, deg] # sequence 25
# CALIB = [-2.54, 0., 0.] # [m, m, deg]

# Split
if IS_UBUNTU:
    SPLITTER = '/' # Ubuntu
else:
    SPLITTER = '\\' # Window

# Delay
DELAY_FRAME = 0

# State Global
SG_NORMAL = 0
SG_START_LABELING = 1

# State Local
# Labeling Box
SL_START_LABELING = 0
SL_CLICK_CENTER = 1
SL_CLICK_FRONT = 2
SL_END_LABELING = 3

# Path
# PATH_SEQ = None
PATH_SEQ = 'E:\\radar_bin_lidar_bag_files\\generated_files'

PATH_IMG_G = osp.join(BASE_DIR, 'resources', 'imgs', 'prevg.png')
PATH_IMG_L = osp.join(BASE_DIR, 'resources', 'imgs', 'prevl.png')

PATH_IMG_F = osp.join(BASE_DIR, 'resources', 'imgs', 'prevf.png')
PATH_IMG_B = osp.join(BASE_DIR, 'resources', 'imgs', 'prevb.png')

# Font
FONT = 'Times New Roman'
FONT_SIZE = 10

# Image Size
W_BEV = 1280
H_BEV = 800
W_CAM = 320
H_CAM = 240

# Button Type
BT_LEFT = 0
BT_RIGHT = 1
BT_MIDDLE = 2

RANGE_Z = [-3, 3] # [m]

# Class
LIST_CLS_NAME = [
    'Sedan',
    'Bus or Truck',
    'Motorcycle',
    'Bicycle',
    'Pedestrian',
    'Pedestrian Group',
    'Bicycle Group',
]

# BGR
LIST_CLS_COLOR = [
    [0,255,0],
    [0,50,255],
    [0,0,255],
    [0,200,255],
    [255,0,0],
    [255,0,100],
    [255,200,0],
]

LIST_Z_CEN_LEN = [
    [-1.5,0.95],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
    [-1.5,1.5],
]

# Drawing
LINE_WIDTH = 2

# Front & Beside Image
RANGE_Y_FRONT = [-4,4]  # [m]
RANGE_Z_FRONT = [-8,8]  # [m]
IMG_SIZE_YZ = [300,150] # [pixel]
M_PER_PIX_YZ = 16./300.

RANGE_X_FRONT = [-8,8]  # [m]
RANGE_Z_FRONT = [-8,8]  # [m]
IMG_SIZE_XZ = [300,300] # [pxiel]
M_PER_PIX_XZ = 16./300.
