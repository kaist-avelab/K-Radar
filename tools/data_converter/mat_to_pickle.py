import os
import numpy as np
from scipy.io import loadmat
from glob import glob
import os.path as osp
from tqdm import tqdm
import pickle
import gzip

### Here to change ### 
LIST_DIR = ['/media/donghee/HDD_1/Radar_Data_Examples_2'] # make this to one sequence
GEN_DIR = '/media/donghee/HDD_2/KRadar_Pickle'
### Here to change ###

def save_name_as_seq_in_pickle(path_dir, name_seq):
    print(f'* Start to read seqeunce {name_seq} ...')
    path_save = osp.join(GEN_DIR, name_seq)
    os.makedirs(path_save, exist_ok=True)
    path_tes_dir = osp.join(path_dir, name_seq, 'radar_tesseract')
    list_files = sorted(os.listdir(path_tes_dir))
    for rdr_tes in tqdm(list_files):
        path_tesseract = osp.join(path_tes_dir, rdr_tes)
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        file_num = rdr_tes.split('.')[0].split('_')[1]
        with open(osp.join(path_save,f'rdr_{file_num}.bin'), 'wb') as f:
            pickle.dump(arr_tesseract,f)


if __name__ == '__main__':
    path_dir = LIST_DIR[0]
    list_name_seq = os.listdir(path_dir)
    for name_seq in list_name_seq:
        save_name_as_seq_in_pickle(path_dir, name_seq)
