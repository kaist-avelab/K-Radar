'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import open3d as o3d
import numpy as np
import sys
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from utils.kitti_eval import kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.util_pipeline import read_imageset_file

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

# Preliminaries
# Change variables in get_official_eval_result in kitti_eval/eval.py
# overlap_hard = np.array([[0.98, 0.98, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
#                          [0.98, 0.98, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
#                          [0.98, 0.98, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
# overlap_mod = np.array([[0.95, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                         [0.95, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                         [0.95, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
# overlap_easy = np.array([[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
#                          [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
#                          [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])

### Here to change ###
path_root = './tools/eval_checker/temp'
dir_label = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL'
class TempPipeline(object):
    def __init__(self):
        self.val_keyword = {
            'Sedan': 'sed',
            'Bus or Truck': 'bus',
        }
        self.dict_cls_id_to_name = {
            1: 'Sedan',
            2: 'Bus or Truck',
        }
        self.dict_cls_name_to_id = {
            'Sedan': 1,
            'Bus or Truck': 2,
        }
### Here to change ###

### Function to be revised ###
def dict_datum_to_kitti(p_pline, dict_item):
    '''
    * Assuming batch size as 1
    '''
    list_kitti_pred = []
    list_kitti_gt = []
    dict_val_keyword = p_pline.val_keyword # without empty space for cls name

    # considering gt
    header_gt = '0.00 0 0 50 50 150 150'
    for idx_gt, label in enumerate(dict_item['label'][0]): # Assuming batch size as 1
        cls_name, cls_idx, (xc, yc, zc, rz, xl, yl, zl), _ = label
        xc, yc, zc, rz, xl, yl, zl = np.round(xc, 2), np.round(yc, 2), np.round(zc, 2), np.round(rz, 2), np.round(xl, 2), np.round(yl, 2), np.round(zl, 2),
        cls_val_keyword = dict_val_keyword[cls_name]
        # print(cls_val_keyword)

        # (1) OpenPCDet version
        # box_centers = str(-yc) + ' ' + str(-zc) + ' ' + str(xc)  # x, y, z
        # box_dim = str(xl) + ' ' + str(zl) + ' ' + str(yl)  # height, width, length
        # str_rot = str(-rz-np.pi/2.)

        # (2)
        # box_centers = str(xc) + ' ' + str(zc) + ' ' + str(yc) # xcam, ycam, zcam
        # box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl) # height(ycaml), width(xcaml), length(zcaml)
        # str_rot = str(rz)
        
        # origin
        box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc) # xcam, ycam, zcam
        box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl) # height(ycaml), width(xcaml), length(zcaml)
        str_rot = str(rz)

        kitti_gt = cls_val_keyword + ' ' + header_gt  + ' ' + box_dim  + ' ' + box_centers + ' ' + str_rot
        list_kitti_gt.append(kitti_gt)
    
    if dict_item['pp_num_bbox'] == 0: # empty prediction (should consider for lpc)
        # KITTI Example: Car -1 -1 -4.2780 668.7884 173.1068 727.8801 198.9699 1.4607 1.7795 4.5159 5.3105 1.4764 43.1853 -4.1569 0.9903
        kitti_dummy = 'dummy -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0'
        list_kitti_pred.append(kitti_dummy)
    else:
        list_pp_cls = dict_item['pp_cls']
        header_pred = '-1 -1 0 50 50 150 150'
        for idx_pred, pred_box in enumerate(dict_item['pp_bbox']):
            score, xc, yc, zc, xl, yl, zl, rot = pred_box
            cls_id = list_pp_cls[idx_pred]
            cls_name = p_pline.dict_cls_id_to_name[cls_id]
            cls_val_keyword = dict_val_keyword[cls_name]
            
            # (1) OpenPCDet version
            # box_centers = str(-yc) + ' ' + str(-zc) + ' ' + str(xc)  # x, y, z
            # box_dim = str(xl) + ' ' + str(zl) + ' ' + str(yl)  # height, width, length
            # str_rot = str(-rot-np.pi/2.)

            # (2)
            # box_centers = str(xc) + ' ' + str(zc) + ' ' + str(yc)
            # box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl)
            # str_rot = str(rot)
            
            # origin
            box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc)
            box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl)
            str_rot = str(rot)

            str_score = str(score)
            kitti_pred = cls_val_keyword + ' ' + header_pred  + ' ' + box_dim + ' ' + box_centers + ' ' + str_rot + ' ' + str_score
            list_kitti_pred.append(kitti_pred)

    dict_item['kitti_pred'] = list_kitti_pred
    dict_item['kitti_gt'] = list_kitti_gt
    
    return dict_item
### Function to be revised ###

def create_cylinder_mesh(radius, p0, p1, color=[1, 0, 0]):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(np.array(p1)-np.array(p0)))
    cylinder.paint_uniform_color(color)
    frame = np.array(p1) - np.array(p0)
    frame /= np.linalg.norm(frame)
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(frame[2]), np.arctan2(-frame[0], frame[1]), 0))
    cylinder.rotate(R, center=[0, 0, 0])
    cylinder.translate((np.array(p0) + np.array(p1)) / 2)
    return cylinder

def draw_3d_box_in_cylinder(vis, center, theta, l, w, h, color=[1, 0, 0], radius=0.1, in_cylinder=True):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]])
    corners = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                        [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])
    corners_rotated = np.dot(corners, R.T) + center
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_rotated)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    if in_cylinder:
        for line in lines:
            cylinder = create_cylinder_mesh(radius, corners_rotated[line[0]], corners_rotated[line[1]], color)
            vis.add_geometry(cylinder)
    else:
        vis.add_geometry(line_set)

if __name__ == '__main__':
    IS_VIS = False
    # LIST_CARE_SEQ = ['1', '2', '3']
    LIST_CARE_SEQ = None # care all seqs
    SCORE = 0.3

    ### Save files ###
    os.makedirs(path_root, exist_ok=True)
    len_folders = len(os.listdir(path_root))
    path_root = osp.join(path_root, f'{len_folders}'.zfill(3))

    preds_dir = osp.join(path_root, 'preds')
    labels_dir = osp.join(path_root, 'gts')
    split_path = osp.join(path_root, 'val.txt')
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    ### Save files ###

    ### Load label ###
    list_dict_item = []
    list_dir = os.listdir(dir_label)

    temp_pline = TempPipeline()
    list_care_cls = list(temp_pline.val_keyword.keys())

    for seq_name in list_dir:
        if LIST_CARE_SEQ is not None:
            if seq_name not in LIST_CARE_SEQ:
                continue

        file_names = os.listdir(osp.join(dir_label, seq_name))
        for file_name in file_names:
            path_label = osp.join(dir_label, seq_name, file_name)
            f = open(path_label, 'r')
            lines = f.readlines()
            f.close()
            list_tuple_objs = []
            deg2rad = np.pi/180.

            header = (lines[0]).rstrip('\n')
            try:
                temp_idx, tstamp = header.split(', ')
            except: # line breaking error for v2_0
                _, header_prime, line0 = header.split('*')
                header = '*' + header_prime
                temp_idx, tstamp = header.split(', ')
                lines.insert(1, '*'+line0)
                lines[0] = header
            
            for line in lines[1:]:
                list_vals = line.rstrip('\n').split(', ')
                idx_p = int(list_vals[1])
                cls_name = (list_vals[2])
                x = float(list_vals[3])
                y = float(list_vals[4])
                z = float(list_vals[5])
                th = float(list_vals[6])*deg2rad
                l = 2*float(list_vals[7])
                w = 2*float(list_vals[8])
                h = 2*float(list_vals[9])
                if cls_name not in list_care_cls:
                    continue
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p)))
            
            if len(list_tuple_objs) == 0:
                continue
            
            rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
            tstamp = tstamp.split('=')[1]
            dict_item = dict()
            dict_item['label'] = list_tuple_objs
            list_dict_item.append(dict_item)
    ### Load label ###

    ### Process dicts ###
    for frame_idx, dict_item in enumerate(tqdm(list_dict_item)):
        list_tuple_objs = dict_item['label']

        list_gt = []
        list_pp_bbox = []
        list_pp_cls = []
        for idx_obj, tuple_obj in enumerate(list_tuple_objs):
            cls_name, (x, y, z, th, l, w, h), _ = tuple_obj
            
            cls_id = temp_pline.dict_cls_name_to_id[cls_name]
            list_gt.append([cls_name, cls_id, (x, y, z, th, l, w, h), idx_obj])

            # same as gt
            list_pp_bbox.append([SCORE, x, y, z, l, w, h, th])

            # np.pi/2. rotated (same as xl and yl changed)
            # list_pp_bbox.append([SCORE, x, y, z, l, w, h, th+np.pi/2.])
            
            list_pp_cls.append(cls_id)

        frame_name = f'{frame_idx}'.zfill(6)
        
        dict_item['label'] = []
        dict_item['label'].append(list_gt)
        dict_item['pp_bbox'] = list_pp_bbox
        dict_item['pp_cls'] = list_pp_cls
        dict_item['pp_num_bbox'] = len(list_pp_bbox)
        
        dict_item = dict_datum_to_kitti(temp_pline, dict_item)
        for idx_label, label in enumerate(dict_item['kitti_gt']):
            mode = 'w' if idx_label == 0 else 'a'
            with open(labels_dir + '/' + frame_name + '.txt', mode) as f:
                f.write(label+'\n')
        if len(dict_item['kitti_pred']) == 0:
            with open(preds_dir + '/' + frame_name + '.txt', mode) as f:
                f.write('\n')
        else:
            for idx_pred, pred in enumerate(dict_item['kitti_pred']):
                mode = 'w' if idx_pred == 0 else 'a'
                with open(preds_dir + '/' + frame_name + '.txt', mode) as f:
                    f.write(pred+'\n')
        mode = 'w' if frame_idx == 0 else 'a'
        with open(split_path, mode) as f:
            f.write(frame_name + '\n')
    ### Process dicts ###

    ### Visualization ###
    if IS_VIS:
        for idx_dict, dict_item in enumerate(list_dict_item):
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            for obj in dict_item['label'][0]:
                cls_name, cls_idx, (x, y, z, th, l, w, h), _ = obj
                draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[1.,0.,0.], radius=0.05)

            for obj in dict_item['pp_bbox']:
                score, x, y, z, l, w, h, th = obj
                draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[0.,1.,0.], radius=0.05)
            
            vis.run()
            vis.destroy_window()
    ### Visualization ###

    ### Evaluation ###
    preds_dir = osp.join(path_root, 'preds')
    labels_dir = osp.join(path_root, 'gts')
    split_path = osp.join(path_root, 'val.txt')

    dt_annos = kitti.get_label_annos(preds_dir)
    val_ids = read_imageset_file(split_path)
    # print(val_ids)
    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
    # print(dt_annos)
    # print(gt_annos)

    list_metrics = []
    list_results = []
    for idx_cls_val in [0, 1]: # sed, bus
        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
        print(dict_metrics)
    ### Evaluation ###
