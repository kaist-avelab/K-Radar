'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import os.path as osp
import numpy as np
import open3d as o3d
import torch
import random
import nms

'''
* Change skip_line & n_attr based on pcd format
'''

SEED = 1
FOLDER_PCD = '/media/donghee/HDD_0/Radar_Film/gen_files/44/os2-128'
CONFIDENCE_THR = 0.05
PATH_CONFIG = './configs/cfg_PVRCNNPP.yml'
PATH_MODEL = './pretrained/PVRCNNPP_wide_29.pt'
IS_NMS_OUTPUT = True

def get_ldr64(path_pcd, \
              inside_ldr64=True, is_calib=True, \
              skip_line=12, n_attr=5, calib_vals=[-2.54, 0.3, 0.7]):
    with open(path_pcd, 'r') as f:
        lines = [line.rstrip('\n') for line in f][skip_line:]
        pc_lidar = [point.split() for point in lines]
        f.close()

    pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, n_attr)

    if inside_ldr64:
        pc_lidar = pc_lidar[np.where(
            (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
            (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]
    
    if is_calib:
        n_pts, _ = pc_lidar.shape
        calib_vals = np.array(calib_vals).reshape(-1,3).repeat(n_pts, axis=0)
        pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals
    
    return pc_lidar

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

def get_nms_results(pred_boxes, pred_scores, pred_labels, nms_thr=0.1, is_cls_agnostic=False):
    if is_cls_agnostic:
        for idx_cls in np.unique(pred_labels):
            pass # TODO
    else:
        xy_xlyl_th = np.concatenate((pred_boxes[:,0:2], pred_boxes[:,3:5], pred_boxes[:,6:7]), axis=1)
        list_tuple_for_nms = [[(x, y), (xl, yl), th] for (x, y, xl, yl, th) in xy_xlyl_th.tolist()]
        ind = nms.rboxes(list_tuple_for_nms, pred_scores.tolist(), nms_threshold=nms_thr)
        return pred_boxes[ind], pred_scores[ind], pred_labels[ind]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For executing the '__main__' directly
try:
    from utils.util_config import cfg, cfg_from_yaml_file
except:
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from utils.util_config import cfg, cfg_from_yaml_file
    from models.skeletons import build_skeleton

if __name__ == '__main__':
    set_random_seed(SEED)

    cfg = cfg_from_yaml_file(PATH_CONFIG, cfg)
    
    ### Build model ###
    model = build_skeleton(cfg).cuda()
    pt_dict_model = torch.load(PATH_MODEL)
    model.load_state_dict(pt_dict_model, strict=False)
    model.eval()
    ### Build model ###
    
    file_list = sorted(os.listdir(FOLDER_PCD))

    for file_name in file_list:
        path_pcd = os.path.join(FOLDER_PCD, file_name)

        ### Build dictionary inputs (based on getitem & collate_fn) ###
        pc_lidar = get_ldr64(path_pcd)
        # ROI filtering
        x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
        pc_lidar = pc_lidar[np.where(
            (pc_lidar[:, 0] > x_min) & (pc_lidar[:, 0] < x_max) &
            (pc_lidar[:, 1] > y_min) & (pc_lidar[:, 1] < y_max) &
            (pc_lidar[:, 2] > z_min) & (pc_lidar[:, 2] < z_max))]
        # degradation with random choice (128ch -> 64ch)
        # selected_idx = np.random.choice(range(len(pc_lidar)), int(len(pc_lidar)*0.5), replace=False)
        # pc_lidar = pc_lidar[selected_idx,:]
        # getitem & collate_fn
        dict_input = dict()
        list_pc_lidar = torch.from_numpy(pc_lidar).float() # .cuda()
        dict_input['ldr64'] = list_pc_lidar
        batch_indices = torch.full((len(pc_lidar),), 0) # .cuda()
        dict_input['batch_indices_ldr64'] = batch_indices
        dict_input['batch_size'] = 1
        dict_input['gt_boxes'] = torch.zeros((1,1,8))
        ### Build dictionary inputs (based on getitem & collate_fn) ###

        dict_output = model(dict_input)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
        vis.add_geometry(pcd)

        ### Visualize based on pred_dicts & pred_labels ###
        pred_dicts = dict_output['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        if IS_NMS_OUTPUT:
            pred_boxes, pred_scores, pred_labels = get_nms_results(pred_boxes, pred_scores, pred_labels)
        
        for idx_pred in range(len(pred_labels)):
            x, y, z, l, w, h, th = pred_boxes[idx_pred]
            score = pred_scores[idx_pred]
            
            draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[0., 0., 0.], radius=0.05)
            # if score > CONFIDENCE_THR:
            #     cls_idx = pred_labels[idx_pred]
            #     draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[0., 0., 0.], radius=0.05)
            # else:
            #     continue
        ### Visualize based on pred_dicts & pred_labels ###

        vis.run()
        vis.destroy_window()
