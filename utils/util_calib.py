'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
* comment: calibration of LiDAR & Camera
'''

import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

__all__ = [
    'get_matrices_from_dict_calib',
    'show_projected_point_cloud',
    'save_calibration_matrix_in_npy',
]

dict_front0 = dict(
    img_size_w=1280,
    img_size_h=720,
    fx=567.720776478944,
    fy=577.2136917114258,
    px=628.72078,
    py=369.30687,
    k1=-0.028873818,
    k2=0.0006023302,
    k3=0.0039573087,
    k4=-0.0050471763,
    k5=0.0,
    roll_ldr2cam=89.61884714, # -0.4
    pitch_ldr2cam=0.59970393, # 1.8
    yaw_ldr2cam=88.19990136, # 0.6
    x_ldr2cam=-0.18,
    y_ldr2cam=-0.12,
    z_ldr2cam=0.19,
)

def get_rpy_from_rotation_matrix(rotation_matrix):
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('zyx', degrees=True)
    return euler_angles

def get_rpy_reflecting_rotation_default_in_ui_labeling(rotation_matrix):
    rot_default = [[0.0, -1.0,  0.0],
                   [0.0,  0.0, -1.0],
                   [1.0,  0.0,  0.0]]
    new_rot = rotation_matrix@rot_default
    rotation = R.from_matrix(new_rot)
    new_rpy = rotation.as_euler('zyx', degrees=True)
    return new_rpy

def get_matrices_from_dict_calib(dict_calib=dict_front0):
    img_size = (dict_calib['img_size_w'], dict_calib['img_size_h'])
    intrinsics = np.array([
        [dict_calib['fx'], 0.0, dict_calib['px']],
        [0.0, dict_calib['fy'], dict_calib['py']],
        [0.0, 0.0, 1.0]
    ])
    distortion = np.array([
        dict_calib['k1'], dict_calib['k2'], dict_calib['k3'], \
        dict_calib['k4'], dict_calib['k5']
    ]).reshape((-1,1))

    # L to C
    yaw_ldr2cam = dict_calib['yaw_ldr2cam']
    pitch_ldr2cam = dict_calib['pitch_ldr2cam']
    roll_ldr2cam = dict_calib['roll_ldr2cam']
    r_ldr2cam = (R.from_euler('zyx', [yaw_ldr2cam, pitch_ldr2cam, roll_ldr2cam], degrees=True)).as_matrix()

    x_ldr2cam = dict_calib['x_ldr2cam']
    y_ldr2cam = dict_calib['y_ldr2cam']
    z_ldr2cam = dict_calib['z_ldr2cam']
    T_ldr2cam = np.concatenate([r_ldr2cam, np.array([x_ldr2cam,y_ldr2cam,z_ldr2cam]).reshape(-1,1)], axis=1)
    # L to C
    
    return img_size, intrinsics, distortion, T_ldr2cam

def show_projected_point_cloud(img, pcd, list_params, undistort=True):
    img_size, intrinsics, distortion, T_ldr2cam = list_params
    img_process = img

    if undistort:
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        
        for j in range(3):
            for i in range(3):
                intrinsics[j,i] = ncm[j, i]
        
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1)
        img_process = cv2.remap(img_process, map_x, map_y, cv2.INTER_LINEAR)

    ### Scaling ###
    # scale_x = 0.2
    # scale_y = 0.5
    # img_process = cv2.resize(img_process, (0,0), fx=scale_x, fy=scale_y)
    # print(img_process.shape)
    # intrinsics[0,0] = intrinsics[0,0]*scale_x
    # intrinsics[0,2] = intrinsics[0,2]*scale_x
    # intrinsics[1,1] = intrinsics[1,1]*scale_y
    # intrinsics[1,2] = intrinsics[1,2]*scale_y
    ### Scaling ###

    T_cam2pix = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0)
    # * A: [fx 0 px 0], B: [fx 0 0 px] -> A is right one (d*px, d*py) [m] should be translation
    # *    [fy 0 py 0]     [fy 0 0 py]
    # *    [0  0  1 0]     [0  0 1  0]
    # *    [0  0  0 1]     [0  0 0  1]
    
    T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0)
    T_ldr2pix = T_cam2pix@T_ldr2cam

    # pcd = pcd[np.where(pcd[:,0]>0)]
    pc_ldr = (np.insert(pcd[:,:3], 3, [1], axis=1)).T
    pc_cam = T_ldr2pix@pc_ldr
    pc_cam[:2,:] /= pc_cam[2,:]
    
    img_process = np.flip(img_process, axis=2) # bgr to rgb
    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    img_h,img_w,_ = img_process.shape
    plt.axis([0,img_w,img_h,0])
    plt.imshow(img_process)
    pc_cam = (pc_cam.T)[:,:3]
    pc_cam = pc_cam[np.where(
        (pc_cam[:,0]>=0) & (pc_cam[:,0]<img_w) &
        (pc_cam[:,1]>=0) & (pc_cam[:,1]<img_h) &
        (pc_cam[:,2]>3))]
    
    plt.scatter(pc_cam[:,0],pc_cam[:,1],c=1/pc_cam[:,2],cmap='rainbow_r',alpha=0.2,s=1.5)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

def save_calibration_matrix_in_npy(key_cam, list_params, undistort=True, dir_save='./resources/cam_calib/T_npy'):
    img_size, intrinsics, distortion, T_ldr2cam = list_params

    if undistort:
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics[j,i] = ncm[j, i]

    T_cam2pix = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0)
    # * A: [fx 0 px 0], B: [fx 0 0 px] -> A is right one (d*px, d*py) [m] should be translation
    # *    [fy 0 py 0]     [fy 0 0 py]
    # *    [0  0  1 0]     [0  0 1  0]
    # *    [0  0  0 1]     [0  0 0  1]
    
    T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0)
    # T_ldr2pix = T_cam2pix@T_ldr2cam

    path_T_cam2pix = os.path.join(dir_save, f'T_cam2pix_{key_cam}.npy')
    path_T_ldr2cam = os.path.join(dir_save, f'T_ldr2cam_{key_cam}.npy')
    
    print(T_cam2pix)
    print(T_ldr2cam)
    
    np.save(path_T_cam2pix, T_cam2pix)
    np.save(path_T_ldr2cam, T_ldr2cam)

if __name__ == '__main__':
    ### Get new calib params reflecting tr_default in ui_labeling ###
    DIR_CALIB = './resources/cam_calib/common'
    DIR_SAVE_CALIB = './resources/cam_calib/new'
    IMG_SIZE = (1280, 720)

    os.makedirs(DIR_SAVE_CALIB, exist_ok=True)

    for yml_file_name in os.listdir(DIR_CALIB):
        path_calib = os.path.join(DIR_CALIB, yml_file_name)
        print(yml_file_name)

        with open(path_calib, 'r') as yml_file:
            dict_calib = yaml.safe_load(yml_file)
            roll_ldr2cam, pitch_ldr2cam, yaw_ldr2cam = dict_calib['roll_ldr2cam'], dict_calib['pitch_ldr2cam'], dict_calib['yaw_ldr2cam']
            Rot = (R.from_euler('zyx', [yaw_ldr2cam, pitch_ldr2cam, roll_ldr2cam], degrees=True)).as_matrix()
            yaw_new, pitch_new, roll_new = get_rpy_reflecting_rotation_default_in_ui_labeling(Rot)
            dict_calib['roll_ldr2cam'] = float(roll_new)
            dict_calib['pitch_ldr2cam'] = float(pitch_new)
            dict_calib['yaw_ldr2cam'] = float(yaw_new)
            dict_calib['img_size_w'] = IMG_SIZE[0]
            dict_calib['img_size_h'] = IMG_SIZE[1]
            dict_calib.pop('cam_number')

            with open(os.path.join(DIR_SAVE_CALIB, yml_file_name), 'w') as yml_file_save:
                yaml.dump(dict_calib, yml_file_save, default_flow_style=False)
