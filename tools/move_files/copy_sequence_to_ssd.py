import os
import os.path as osp
import shutil
from tqdm import tqdm
import numpy as np

LIST_DIR = ['/media/ave/HDD_4_1/radar_bin_lidar_bag_files/generated_files', \
            '/media/ave/e95e0722-32a4-4880-a5d5-bb46967357d6/radar_bin_lidar_bag_files/generated_files', \
            '/media/ave/4f089d0e-7b60-493d-aac7-86ead9655194/radar_bin_lidar_bag_files/generated_files'] # '/media/ave/HDD_4_1/gen_2to5', 

LIST_SEQS = [['1~31'], ['32~58']]
LIST_WHERE_TO_COPY = ['/media/ave/31_SSD/radar_bin_lidar_bag_files/generated_files', '/media/ave/34_SSD/radar_bin_lidar_bag_files/generated_files']

LIST_MOVE_FOLDER = ['cam-front', 'cam-left', 'cam-right', 'cam-rear', 'info_calib', 'info_label', 'lidar_bev_image', 'os2-64', 'radar_bev_image', 'time_info']
LIST_NEW_FOLDER = ['info_matching', 'info_label_revised']
LIST_MOVE_FILE = ['description.txt']

if __name__ == '__main__':
    ### Pre-processing ###
    LIST_PROCESSED_SEQS = [[] for _ in range(len(LIST_SEQS))]
    for idx_list, temp_list_seqs in enumerate(LIST_SEQS):
        for temp_seq in temp_list_seqs:
            if '~' in temp_seq:
                st_idx, end_idx = temp_seq.split('~')
                list_range = np.arange(int(st_idx), int(end_idx)+1)
                list_range = list(map(lambda x: f'{x}', list_range))
                LIST_PROCESSED_SEQS[idx_list].extend(list_range)
            else:
                LIST_PROCESSED_SEQS[idx_list].append(temp_seq)
    ### Pre-processing ###


    for dir_path in LIST_DIR:
        list_seqs = os.listdir(dir_path)

        for temp_seq in tqdm(list_seqs):
            for idx_list in range(len(LIST_PROCESSED_SEQS)):

                # if seqs exist for the list seq
                if temp_seq in LIST_PROCESSED_SEQS[idx_list]:
                    os.makedirs(osp.join(LIST_WHERE_TO_COPY[idx_list], temp_seq))
                    # copy folder
                    for move_folder in LIST_MOVE_FOLDER:
                        shutil.copytree(osp.join(dir_path, temp_seq, move_folder), osp.join(LIST_WHERE_TO_COPY[idx_list], temp_seq, move_folder))

                    for new_folder in LIST_NEW_FOLDER:
                        os.makedirs(osp.join(LIST_WHERE_TO_COPY[idx_list], temp_seq, new_folder), exist_ok=True)

                    for move_file in LIST_MOVE_FILE:
                        shutil.copy(osp.join(dir_path, temp_seq, move_file), osp.join(LIST_WHERE_TO_COPY[idx_list], temp_seq, move_file))
