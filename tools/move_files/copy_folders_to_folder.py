'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os
import os.path as osp
import shutil
from tqdm import tqdm

### Change here ###
LIST_DIR = ['/media/oem/HDD_3_1/radar_bin_lidar_bag_files/generated_files','/media/oem/HDD_3_2/radar_bin_lidar_bag_files/generated_files','/media/oem/data_21/radar_bin_lidar_bag_files/generated_files']
WANT_TO_COPY = 'sparse_cube_dop'
WHERE_TO_COPY = '/media/oem/9_SSD/SPARSE_DOP_CUBES'
### Change here ###

if __name__ == '__main__':
    for dir_seqs in LIST_DIR:
        list_seqs = os.listdir(dir_seqs)
        
        for seq in tqdm(list_seqs):
            copy_dir = osp.join(dir_seqs, seq, WANT_TO_COPY)
            target_dir = osp.join(WHERE_TO_COPY, seq, WANT_TO_COPY)
            os.makedirs(osp.join(WHERE_TO_COPY, seq), exist_ok=True)
            shutil.copytree(copy_dir, target_dir)
