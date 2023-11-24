'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import open3d as o3d
import numpy as np
import torch
import sys

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from utils.kitti_eval import kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.util_pipeline import read_imageset_file

### Here to change ###
# check tp, fp, fn in eval.py
# Set num_parts of eval_class function in eval.py as 1 
path_root = './tools/eval_checker/temp'
list_frames = ['000000']#, '000001']
dict_gt = {
    '000000': [
        ['Sedan', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
        # ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
        # ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
            ],
    # '000001': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000002': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000003': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000004': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000005': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000006': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000007': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000008': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000009': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000010': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000011': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000012': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000013': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
    # '000014': [
    #     ['Sedan', 1, (10.1, 20.-30, 1.72, 1.54, 2.57, 1.2, 2.4), 0],
    #     ['Sedan', 1, (20.2, -20.1-30, 1.51, 0.3, 2.52, 1.2, 2.4), 1],
    #         ],
    # '000015': [
    #     ['Bus or Truck', 2, (10.2, 20.1, 1.71, 1.55, 8.24, 4.2, 4.56), 0],
    #     ['Sedan', 1, (20.1, -20.2, 1.52, 0.31, 2.52, 1.2, 2.4), 1],
    # ],
}
dict_pp_bbox = {
    '000000': [
        [1.0, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
        # [1.0, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
        # [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    ],
    # '000001': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000002': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000003': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000004': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000005': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000006': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000007': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000008': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000009': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000010': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000011': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000012': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000013': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000014': [
    #     [0.81, 10, 20, 1.7, 2.57, 1.2, 2.4, 1.54],
    #     [0.71, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
    # '000015': [
    #     [0.81, 10, 20, 1.7, 8.24, 4.2, 4.56, 1.54],
    #     [0.81, 20, -20, 1.5, 2.52, 1.2, 2.4, 0.3],
    # ],
}
dict_pp_cls = {
    '000000': [1],
    # '000001': [2, 1],
    # '000002': [1, 1],
    # '000003': [2, 1],
    # '000004': [1, 1],
    # '000005': [2, 1],
    # '000006': [1, 1],
    # '000007': [2, 1],
    # '000008': [1, 1],
    # '000009': [2, 1],
    # '000010': [1, 1],
    # '000011': [2, 1],
    # '000012': [1, 1],
    # '000013': [2, 1],
    # '000014': [1, 1],
    # '000015': [2, 1],
}
### Here to change ###

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

        # (1)
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
            
            # (1)
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

    ### Process dicts ###
    list_dicts = []
    temp_pline = TempPipeline()
    for frame_idx, frame_name in enumerate(list_frames):
        dict_item = dict()
        list_gt = dict_gt[frame_name]
        list_pp_bbox = dict_pp_bbox[frame_name]
        list_pp_cls = dict_pp_cls[frame_name]
        
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
        list_dicts.append(dict_item)
    ### Process dicts ###

    ### Visualization ###
    if IS_VIS:
        for idx_dict, dict_item in enumerate(list_dicts):
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
    for idx_cls_val in [0]: #, 1]: # sed, bus
        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
        print(dict_metrics)
    ### Evaluation ###
