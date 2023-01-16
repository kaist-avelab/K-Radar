import os
import os.path as osp
import sys
import shutil
import numpy as np

### Here to change ###
LIST_PATH_DATASET = ['/media/data3/radar_bin_lidar_bag_files/generated_files', '/media/data4/radar_bin_lidar_bag_files/generated_files', '/media/data2/radar_bin_lidar_bag_files/generated_files']
### Here to change ###

def get_dict_key_seq_val_label(path_datset):
    dict_ret = dict()
    list_seq_num = sorted(os.listdir(path_datset))

    for seq in list_seq_num:
        list_labels_per_seq = sorted(os.listdir(osp.join(path_datset, seq, 'info_label')))
        dict_ret[seq] = list_labels_per_seq
    
    return dict_ret

def sort_list_char_via_int_order(list_sort):
    ''''
    * e.g., in  : ['9', '10', '1', '15', '8', '100']
    *       out : ['1', '8', '9', '10', '15', '100']
    '''
    list_ret = sorted(list(map(lambda x: int(x), list_sort)))
    list_ret = list(map(lambda x: f'{x}', list_ret))

    return list_ret

def add_seq_and_list_label_to_txt(txt, seq, list_label):
    txt_ret = txt
    for label in list_label:
        txt_ret += f'{seq},{label}\n'
    return txt_ret

def divide_to_consecutive_quater(list_to_be):
    num_val = len(list_to_be)

    idx_0 = int(num_val*0.25)
    idx_1 = int(num_val*0.50)
    idx_2 = int(num_val*0.75)
    idx_3 = num_val

    return list_to_be[:idx_0], list_to_be[idx_0:idx_1], \
            list_to_be[idx_1:idx_2], list_to_be[idx_2:idx_3]

def get_txt_train_and_test(list_path_dataset):
    dict_seq = dict()
    for path_dataset in list_path_dataset:
        dict_seq.update(get_dict_key_seq_val_label(path_dataset))
    
    sorted_keys = sort_list_char_via_int_order(list(dict_seq.keys()))
    
    txt_train = ''
    txt_test = ''

    print(f'sequences = {sorted_keys}')
    for seq in sorted_keys:
        list_0, list_1, list_2, list_3 = divide_to_consecutive_quater(dict_seq[seq])
        txt_train = add_seq_and_list_label_to_txt(txt_train, seq, list_0)
        txt_test  = add_seq_and_list_label_to_txt(txt_test,  seq, list_1)
        txt_train = add_seq_and_list_label_to_txt(txt_train, seq, list_2)
        txt_test  = add_seq_and_list_label_to_txt(txt_test,  seq, list_3)
    
    return txt_train[:-1], txt_test[:-1] # delete '\n'

def save_to_path(txt, path_file):
    f = open(path_file, 'w')
    f.write(txt)
    f.close()

if __name__ == '__main__':
    txt_train, txt_test = get_txt_train_and_test(LIST_PATH_DATASET)
    save_to_path(txt_train, './train.txt')
    save_to_path(txt_test, './test.txt')
