'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, Dongin Kim, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, dongin.kim@kaist.ac.kr
* description: revise label for 3D object detection
'''

import os
import os.path as osp
import sys

from glob import glob

class Label_revisor():
    def __init__(self, cfg=None, split='train'):
        super().__init__()
        self.cfg = cfg
        self.list_path_label_1 = [] # a list of dic
        self.list_path_label_2 = [] # a list of dic
        list_path_calib = []

        list_seq = self.cfg.DATASET.DIR.REVISE_LABEL_DIR
        label_rev1 = osp.join(list_seq[0])
        label_rev2 = osp.join(list_seq[1])
        
        list_rev1 = sorted(os.listdir(label_rev1), key=lambda x : int(x.split('_')[0]))
        list_rev2 = sorted(os.listdir(label_rev2), key=lambda x : int(x))
        for i in range(len(list_rev2)):
            label_path_1 = sorted(glob(osp.join(label_rev1, list_rev1[i], '*.txt')))
            label_path_2 = sorted(glob(osp.join(label_rev2, list_rev2[i], '*.txt')))
            self.list_path_label_1.extend(label_path_1)
            self.list_path_label_2.extend(label_path_2)

        for dir_seq in self.cfg.DATASET.DIR.LIST_DIR:
            list_seq = os.listdir(dir_seq)
            for seq in list_seq:
                seq_label_paths = sorted(glob(osp.join(dir_seq, seq, 'info_calib', '*.txt')))
                list_path_calib.extend(seq_label_paths)
        self.list_path_calib = sorted(list_path_calib, key=lambda x : int(x.split('/')[-3]))

    def compare_file_lines(self, file_path1, file_path2):
        with open(file_path1, 'r') as file1:
            lines1 = file1.readlines()
        
        with open(file_path2, 'r') as file2:
            lines2 = file2.readlines()

        if len(lines1) != len(lines2):
            print(f"The number of lines is different in the files:")
            print(f"File path 1: {file_path1}")
            print(f"File path 2: {file_path2}")


    def get_label_bboxes(self, path_label_1, path_label_2, calib_info, base_path_label_2_1):
        with open(path_label_1, 'r') as f1:
            lines_1 = f1.readlines()
            f1.close()

        with open(path_label_2, 'r') as f2:
            lines_2 = f2.readlines()
            f2.close()

        # print('* lines : ', lines)
        line_objects_1 = lines_1[1:]
        line_objects_2 = lines_2[1:]


        path_label_2_1 = os.path.join(base_path_label_2_1, path_label_2.split('/')[-2], path_label_2.split('/')[-1])
        os.makedirs(os.path.dirname(path_label_2_1), exist_ok=True)

        iou = 0
        new_label = []
        origin_label = []
        extra_label = []


        with open(path_label_2_1, 'w') as w2:
            w2.write('{0}'.format(lines_2[0]))
            
            for i in range(len(line_objects_2)):
                best_iou = 0
                best_line = None
                
                for j in range(len(line_objects_1)):
                    line2 = line_objects_2[i].split(',')
                    line1 = line_objects_1[j].split(',')
                    if line1[0] == '#':
                        continue
                    else: 
                        bbox2 = float(line2[3]), float(line2[4]), float(line2[5]), float(line2[7]), float(line2[8]), float(line2[9]) 
                        bbox1 = float(line1[5]), float(line1[6]), float(line1[7]), float(line1[9]), float(line1[10]), float(line1[11]) 
                        iou = self.compute_3d_iou(bbox2, bbox1)
                        if iou > best_iou:
                            best_iou = iou
                            best_line = line1
                
                if best_iou > 0.01:
                    new_label.append(line_objects_2[i])
                    line2.insert(1, best_line[1])
                    w2.write('{0}'.format(','.join(line2)))

                origin_label.append(line_objects_2[i])
            for i in origin_label :
                if i not in new_label :
                    extra_label = i.split(',')
                    if len(extra_label) < 3:
                        continue
                    if extra_label[0] == '#':
                        continue
                    else:
                        extra_label.insert(1, ' L1')
                        w2.write('{0}'.format(extra_label[0]+str(',')+extra_label[1]+str(',')+extra_label[2]+str(',')+extra_label[3]+str(',')+extra_label[4]+str(','))+extra_label[5]+str(',')+extra_label[6]+str(',')+extra_label[7]+str(',')+extra_label[8]+str(',')+extra_label[9]+str(',')+extra_label[10])
        
        revisor.compare_file_lines(path_label_2, path_label_2_1) ### For error check

        return iou

    def compute_3d_iou(self, box1, box2):
        # box - [x, y, z, l, w, h]
        x1, y1, z1, l1, w1, h1 = box1
        x2, y2, z2, l2, w2, h2 = box2
        
        x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = x1 - l1/2, y1 - w1/2, z1 - h1/2, x1 + l1/2, y1 + w1/2, z1 + h1/2
        x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = x2 - l2/2, y2 - w2/2, z2 - h2/2, x2 + l2/2, y2 + w2/2, z2 + h2/2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_zmin = max(z1_min, z2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        inter_zmax = min(z1_max, z2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_depth = max(0, inter_zmax - inter_zmin)
        
        inter_volume = inter_width * inter_height * inter_depth
        
        volume1 = l1 * w1 * h1
        volume2 = l2 * w2 * h2
        
        union_volume = volume1 + volume2 - inter_volume
        
        iou = inter_volume / union_volume if union_volume != 0 else 0
        
        return iou


if __name__ == '__main__':
    import cv2
    import yaml
    from easydict import EasyDict

    path_cfg = './configs/cfg_RTNH_refined_label_revisor.yml'
    label_2_1_path = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility'
    f = open(path_cfg, 'r')
    try:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    f.close()

    revisor = Label_revisor(cfg=cfg)
    label_1_path_list = revisor.list_path_label_1
    label_2_path_list = revisor.list_path_label_2
    calib_path_list = revisor.list_path_calib

    ## 1. revise false label with \n problem * (revise washington label false --> washington label true (e.g., Washington_label/2/00488_00458.txt))
    error_label = []
    for i in range(len(label_1_path_list)):
        calib_index = int(label_1_path_list[i].split('/')[-2].split('_')[0])
        with open(label_2_path_list[i], 'r') as f1:
            lines_1 = f1.readlines()
            f1.close()
        if len(lines_1[0]) > 200:
            with open(label_2_path_list[i], 'w') as w1:
                for i in range(len(lines_1)+1):
                    if i==0:
                        w1.write('{0} \n'.format(lines_1[0].split(',')[0] + ',' +lines_1[0].split(',')[1].split('*')[0]))
                    elif i==1:
                        w1.write('{0}'.format(str('*,')+lines_1[0].split(',')[2]+str(',')+lines_1[0].split(',')[3]+str(',')+lines_1[0].split(',')[4]+str(',')+lines_1[0].split(',')[5]+str(',')+lines_1[0].split(',')[6]+str(',')+lines_1[0].split(',')[7]+str(',')+lines_1[0].split(',')[8]+str(',')+lines_1[0].split(',')[9]+str(',')+lines_1[0].split(',')[10]))
                    else:
                        w1.write('{0}'.format(lines_1[i-1]))
    ## 1. revise false label with \n problem * (revise washington label false --> washington label true (e.g., Washington_label/2/00488_00458.txt))

    ### 2. revise label with ioU (washington label true --> kradar_v_2_1 (using kradar_v_1_1))
    for i in range(len(label_1_path_list)):
        calib_index = int(label_1_path_list[i].split('/')[-2].split('_')[0])
        iou = revisor.get_label_bboxes(label_1_path_list[i], label_2_path_list[i], calib_path_list[calib_index-1], label_2_1_path)
    ### 2. revise label with ioU (washington label true --> kradar_v_2_1 (using kradar_v_1_1))
    