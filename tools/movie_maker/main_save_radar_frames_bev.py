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

from glob import glob
from tqdm import tqdm
from scipy.io import loadmat

### Hyper-params ###
LIST_DIR = ['/media/ave/HDD_4_1/gen_2to5', \
    '/media/ave/HDD_4_1/radar_bin_lidar_bag_files/generated_files',
    '/media/ave/e95e0722-32a4-4880-a5d5-bb46967357d6/radar_bin_lidar_bag_files/generated_files',
    '/media/ave/4f089d0e-7b60-493d-aac7-86ead9655194/radar_bin_lidar_bag_files/generated_files']
PATH_SAVE = '/media/ave/4f089d0e-7b60-493d-aac7-86ead9655194/KRadar_radar_frames'
ROI_X = [0, 0.4, 80]
ROI_Y = [-40, 0.4, 40]
### Hyper-params ###

def get_xy_from_ra_color(ra_in, arr_range_in, arr_azimuth_in, roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_in_deg=False):
    '''
    * args:
    *   roi_x = [min_x, bin_x, max_x] [m]
    *   roi_y = [min_y, bin_y, max_y] [m]
    '''
    
    if len(ra_in.shape) == 2:
        num_range, num_azimuth = ra_in.shape
    elif len(ra_in.shape) == 3:
        num_range, num_azimuth, _ = ra_in.shape

    assert (num_range == len(arr_range_in) and num_azimuth == len(arr_azimuth_in))
    
    ra = ra_in.copy()
    arr_range = arr_range_in.copy()
    arr_azimuth = arr_azimuth_in.copy()

    if is_in_deg:
        arr_azimuth = arr_azimuth*np.pi/180.
    
    min_x, bin_x, max_x = roi_x
    min_y, bin_y, max_y = roi_y

    # with +1
    arr_x = np.linspace(min_x, max_x, int((max_x-min_x)/bin_x)+1)
    arr_y = np.linspace(min_y, max_y, int((max_y-min_y)/bin_y)+1)
    # print(arr_x), print(arr_y)

    max_r = np.max(arr_range)
    min_r = np.min(arr_range)

    max_azi = np.max(arr_azimuth)
    min_azi = np.min(arr_azimuth)

    num_y = len(arr_y)
    num_x = len(arr_x)
    
    arr_yx = np.zeros((num_y, num_x, 3), dtype=np.uint8)

    # print(f'arr_yx shape: {arr_yx.shape}')

    # Inverse warping
    for idx_y, y in enumerate(arr_y):
        for idx_x, x in enumerate(arr_x):
            # bilinear interpolation

            # print(f'y, x = {y}, {x}')
            r = np.sqrt(x**2 + y**2)
            azi = np.arctan2(-y,x) # for real physical azimuth
            # azi = np.arctan2(y,x)
            # print(f'r, azi = {r}, {azi}')
            
            if (r < min_r) or (r > max_r) or (azi < min_azi) or (azi > max_azi):
                continue
            
            try:
                idx_r_0, idx_r_1 = find_nearest_two(r, arr_range)
                idx_a_0, idx_a_1 = find_nearest_two(azi, arr_azimuth)
            except:
                continue

            if (idx_r_0 == -1) or (idx_r_1 == -1) or (idx_a_0 == -1) or (idx_a_1 == -1):
                continue
            
            ra_00 = ra[idx_r_0,idx_a_0,:]
            ra_01 = ra[idx_r_0,idx_a_1,:]
            ra_10 = ra[idx_r_1,idx_a_0,:]
            ra_11 = ra[idx_r_1,idx_a_1,:]

            val = (ra_00*(arr_range[idx_r_1]-r)*(arr_azimuth[idx_a_1]-azi)\
                    +ra_01*(arr_range[idx_r_1]-r)*(azi-arr_azimuth[idx_a_0])\
                    +ra_10*(r-arr_range[idx_r_0])*(arr_azimuth[idx_a_1]-azi)\
                    +ra_11*(r-arr_range[idx_r_0])*(azi-arr_azimuth[idx_a_0]))\
                    /((arr_range[idx_r_1]-arr_range[idx_r_0])*(arr_azimuth[idx_a_1]-arr_azimuth[idx_a_0]))

            arr_yx[idx_y, idx_x] = val

    return arr_yx, arr_y, arr_x

def find_nearest_two(value, arr):
    '''
    * args
    *   value: float, value in arr
    *   arr: np.array
    * return
    *   idx0, idx1 if is_in_arr else -1
    '''
    arr_temp = arr - value
    arr_idx = np.argmin(np.abs(arr_temp))
    
    try:
        if arr_temp[arr_idx] < 0: # min is left
            if arr_temp[arr_idx+1] < 0:
                return -1, -1
            return arr_idx, arr_idx+1
        elif arr_temp[arr_idx] >= 0:
            if arr_temp[arr_idx-1] >= 0:
                return -1, -1
            return arr_idx-1, arr_idx
    except:
        return -1, -1

if __name__ == '__main__':
    list_path_radar = []
    for dir_seq in LIST_DIR:
        list_seq = os.listdir(dir_seq)
        for seq in list_seq:
            seq_radar_paths = sorted(glob(osp.join(dir_seq, seq, 'radar_tesseract', 'tesseract_*.mat')))
            list_path_radar.extend(seq_radar_paths)
    
    for i in range(58):
        os.makedirs(osp.join(PATH_SAVE, f'{i+1}'), exist_ok=True)

    temp_values = loadmat('./resources/info_arr.mat')
    arr_range = temp_values['arrRange']
    deg2rad = np.pi/180.
    arr_azimuth = temp_values['arrAzimuth']*deg2rad
    arr_elevation = temp_values['arrElevation']*deg2rad
    _, num_0 = arr_range.shape
    _, num_1 = arr_azimuth.shape
    _, num_2 = arr_elevation.shape
    arr_range = arr_range.reshape((num_0,))
    arr_azimuth = arr_azimuth.reshape((num_1,))
    arr_elevation = arr_elevation.reshape((num_2,))
    arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
    _, num_3 = arr_doppler.shape
    arr_doppler = arr_doppler.reshape((num_3,))
    arr_azimuth = np.flip(-arr_azimuth)
    arr_elevation = np.flip(-arr_elevation)

    for path_tesseract in tqdm(list_path_radar):
        path_split = path_tesseract.split('/')
        seq_name = path_split[-3]
        file_name = path_split[-1].split('.')[0]+'.png'
        path_save = osp.join(PATH_SAVE, seq_name, file_name)
        
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2)) # in DRAE
        arr_tesseract = np.flip(np.flip(arr_tesseract, axis=2), axis=3)

        rdr_bev = np.mean(np.mean(arr_tesseract, axis=0), axis=2)

        arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)

        height, width = np.shape(rdr_bev)
        # print(height, width)
        # figsize = (1, height/width) if height>=width else (width/height, 1)
        # plt.figure(figsize=figsize)
        plt.clf()
        plt.cla()
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.savefig('./resources/imgs/img_tes_ra.png', bbox_inches='tight', pad_inches=0, dpi=300)
        
        temp_img = cv2.imread('./resources/imgs/img_tes_ra.png')
        temp_row, temp_col, _ = temp_img.shape
        if not (temp_row == height and temp_col == width):
            temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('./resources/imgs/img_tes_ra.png', temp_img_new)

        plt.close()
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
        plt.colorbar()
        plt.savefig('./resources/imgs/plot_tes_ra.png', dpi=300)

        # Polar to Cartesian (Should flip image)
        ra = cv2.imread('./resources/imgs/img_tes_ra.png')
        ra = np.flip(ra, axis=0)

        arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra, arr_range, arr_azimuth, \
            roi_x = ROI_X, roi_y = ROI_Y, is_in_deg=False)
        arr_yx = arr_yx.transpose((1,0,2))
        arr_yx = np.flip(arr_yx, axis=(0,1))
        
        # arr_yx = cv2.resize(arr_yx,(0,0),fx=2,fy=2)
        # cv2.imshow('Cartesian (bbox)', arr_yx)
        # cv2.waitKey(0)

        cv2.imwrite(path_save, arr_yx)

    np.save(osp.join(PATH_SAVE, 'arr_y.npy'), arr_y)
    np.save(osp.join(PATH_SAVE, 'arr_x.npy'), arr_x)
