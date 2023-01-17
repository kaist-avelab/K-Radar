import os
import os.path as osp
import sys
import shutil
import numpy as np

### Here to change ###
PATH_SEQ = '/media/donghee/5_SSD/radar_bin_lidar_bag_files/generated_files/25'
FRAME_DIFFERNECE = 6
X_CALIB = -2.640
Y_CALIB = +0.500
### Here to change ###

def get_dict_and_list_time_info(path_time_info):
    f = open(path_time_info, 'r')
    lines = f.readlines()
    list_str_idx = list(map(lambda x: x.split(', ')[0].split('.')[0].split('_')[1], lines))
    list_str_time = list(map(lambda x: x.split(', ')[1][:-1], lines))
    f.close()

    dict_idx_time = dict()
    for pair in zip(list_str_idx, list_str_time):
        dict_idx_time[pair[0]] = pair[1]
    
    return dict_idx_time, list(map(lambda x: float(x), list_str_time))

if __name__ == '__main__':
    os.makedirs(osp.join(PATH_SEQ, 'info_label_rev'), exist_ok=True)
    list_rev_folder = os.listdir(osp.join(PATH_SEQ, 'info_label_rev'))
    idx_folder = 0
    while True:
        if idx_folder > 99:
            sys.exit()
        idx_folder += 1
        
        if str(idx_folder).zfill(3) in list_rev_folder:
            continue
        else:
            PATH_REV = osp.join(PATH_SEQ, 'info_label_rev', str(idx_folder).zfill(3), 'info_label')
            PATH_REV_CALIB = osp.join(PATH_SEQ, 'info_label_rev', str(idx_folder).zfill(3), 'calib_radar_lidar.txt')
            PATH_REV_BACKUP = osp.join(PATH_SEQ, 'info_label_rev', f'{str(idx_folder).zfill(3)}_backup', 'info_label')
            PATH_REV_CALIB_BACKUP = osp.join(PATH_SEQ, 'info_label_rev', f'{str(idx_folder).zfill(3)}_backup', 'calib_radar_lidar.txt')
            os.makedirs(PATH_REV, exist_ok=True)
            shutil.copytree(osp.join(PATH_SEQ, 'info_label'), PATH_REV_BACKUP)
            if osp.exists(osp.join(PATH_SEQ, 'info_calib', 'calib_radar_lidar.txt')):
                shutil.copy2(osp.join(PATH_SEQ, 'info_calib', 'calib_radar_lidar.txt'), PATH_REV_CALIB_BACKUP)
            break
    
    list_pc = sorted(os.listdir(osp.join(PATH_SEQ, 'os2-64')))
    len_os2_64 = len(list_pc)
    len_lidar_bev_img = len(os.listdir(osp.join(PATH_SEQ, 'lidar_bev_image')))
    assert not (len_os2_64 == len_lidar_bev_img), 'point cloud & point cloud img number is different!'

    list_name_label = sorted(os.listdir(osp.join(PATH_SEQ, 'info_label')))
    len_label = len(list_name_label)
    # assert not (len_label == len_os2_64), 'point cloud & label number is different!'

    list_rar = sorted(os.listdir(osp.join(PATH_SEQ, 'radar_tesseract')))

    print(f'pc # = {len_os2_64}')
    print(f'label # = {len_label}')
    print(f'last pc = {list_pc[-1]}')
    print(f'last label = {list_name_label[-1]}')

    # get timestamp as string, key = str idx, value = str timestamp
    dict_os2, lst_os2 = get_dict_and_list_time_info(osp.join(PATH_SEQ, 'time_info', 'os2-64.txt'))
    dict_os1, lst_os1 = get_dict_and_list_time_info(osp.join(PATH_SEQ, 'time_info', 'os1-128.txt'))
    dict_camf, lst_camf = get_dict_and_list_time_info(osp.join(PATH_SEQ, 'time_info', 'cam-front.txt'))
    dict_camlrr, lst_camlrr = get_dict_and_list_time_info(osp.join(PATH_SEQ, 'time_info', 'cam-left.txt'))

    # print(dict_os2, lst_os2)
    
    num_avail = 0
    for name_label in list_name_label:
        temp_list_str_idx = name_label.split('.')[0].split('_')
        # rar_idx_b4 = int(temp_list_str_idx[0])
        lar_idx_b4 = int(temp_list_str_idx[1])
        
        # if True: # FRAME_DIFFERNECE > 0: # criterion is lidar
        lar_idx = lar_idx_b4
        rar_idx = lar_idx_b4 + FRAME_DIFFERNECE
        
        lar_idx = str(lar_idx).zfill(5)
        rar_idx = str(rar_idx).zfill(5)

        # check if datum exist
        temp_path_rar = osp.join(PATH_SEQ, 'radar_tesseract', f'tesseract_{rar_idx}.mat')
        temp_path_lar = osp.join(PATH_SEQ, 'os2-64', f'os2-64_{lar_idx}.pcd')

        if (not osp.exists(temp_path_rar)) and (not osp.exists(temp_path_lar)):
            print(f'{temp_path_rar} does not exist!')
            print(f'{temp_path_lar} does not exist!')
            continue

        if not osp.exists(temp_path_rar):
            print(f'{temp_path_rar} does not exist!')
            continue

        if not osp.exists(temp_path_lar):
            print(f'{temp_path_lar} does not exist!')
            continue
        
        # make new available labels
        f = open(osp.join(PATH_SEQ, 'info_label', name_label), 'r')
        lines = f.readlines()
        str_time = dict_os2[lar_idx]
        flt_time = float(str_time)

        camf_idx = list(dict_camf.keys())[np.argmin(np.abs(np.array(lst_camf)-flt_time))]
        os1_idx = list(dict_os1.keys())[np.argmin(np.abs(np.array(lst_os1)-flt_time))]
        camlrr_idx = list(dict_camlrr.keys())[np.argmin(np.abs(np.array(lst_camlrr)-flt_time))]
        
        name_file = f'{rar_idx}_{lar_idx}.txt'
        txt_new = f'* idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)={rar_idx}_{lar_idx}_{camf_idx}_{os1_idx}_{camlrr_idx}, timestamp={str_time}'
        # remove 1st line
        if not (len(lines) == 1):
            txt_new += '\n'
            lines = lines[1:]
            for line in lines:
                txt_new += line
            # print(lines)
        # print(txt_new)
        f.close()

        f = open(osp.join(PATH_REV, name_file), 'w')
        f.write(txt_new)
        f.close()
        num_avail += 1
    
    print(f'available labels = {num_avail}')

    ### Calibration ###
    os.makedirs(osp.join(PATH_SEQ, 'info_calib'), exist_ok=True)
    f = open(PATH_REV_CALIB, 'w')
    f.write(f'frame difference,x,y\n{FRAME_DIFFERNECE},{X_CALIB},{Y_CALIB}')
    f.close()
