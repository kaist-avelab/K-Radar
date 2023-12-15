'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
from easydict import EasyDict

from torch.utils.data import Dataset

roi = [0,-15,-2,72,15,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/oem/c886c17a-87f1-42c0-9d3c-831467b43160/Kradar_portion/radar_bin_lidar_bag_files/generated_files'],
        split = ['./resources/split/train.txt', './resources/split/test.txt'],
        revised_label_v1_1 = './tools/revise_label/kradar_revised_label_v1_1',
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
        revised_label_v2_1 = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility',
    ),
    label = { # (consider, logit_idx, rgb, bgr)
        'calib':            True,
        'onlyR':            False,
        'consider_cls':     False,
        'consider_roi':     False,
        'remove_0_obj':     False,
        'Sedan':            [True,  1,  [0, 1, 0],       [0,255,0]],
        'Bus or Truck':     [True,  2,  [1, 0.2, 0],     [0,50,255]],
        'Motorcycle':       [False, -1, [1, 0, 0],       [0,0,255]],
        'Bicycle':          [False, -1, [1, 1, 0],       [0,255,255]],
        'Bicycle Group':    [False, -1, [0, 0.5, 1],     [0,128,255]],
        'Pedestrian':       [False, -1, [0, 0, 1],       [255,0,0]],
        'Pedestrian Group': [False, -1, [0.4, 0, 1],     [255,0,100]],
        'Label':            [False, -1, [0.5, 0.5, 0.5], [128,128,128]],
    },
    label_version = 'v2_0', # ['v1_0', 'v1_1', 'v2_0', v2_1']
    item = dict(calib=True, ldr64=True, ldr128=False, rdr=False, rdr_sparse=True, cam=False),
    calib = dict(z_offset=0.7),
    ldr64 = dict(processed=False, skip_line=13, n_attr=9, inside_ldr64=True, calib=True,),
    rdr = dict(cube=False,),
    rdr_sparse = dict(
        processed=True,
        dir='/media/oem/c886c17a-87f1-42c0-9d3c-831467b43160/Kradar_portion/rdr_sparse_data/rtnh_wider_1p_1'
    ),
    roi = dict(filter=True, xyz=roi, keys=['ldr64', 'rdr_sparse'], check_azimuth_for_rdr=True, azimuth_deg=[-53,53]),
    joint_samples = 5,
)

class KRadarTracking_v1_0(Dataset):
    def __init__(self, cfg=None, split='all'):
        if cfg == None:
            cfg = EasyDict(dict_cfg)
            cfg_from_yaml = False
            self.cfg=cfg
        else:
            cfg_from_yaml = True
            self.cfg=cfg.DATASET

        self.label = self.cfg.label
        self.label_version = self.cfg.get('label_version', 'v2_0')
        self.load_label_in_advance = True if self.label.remove_0_obj else False
        
        self.item = self.cfg.item
        self.calib = self.cfg.calib
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.roi = self.cfg.roi

        self.joint_samples = self.cfg.get('joint_samples', None)
        self.list_dict_item = self.load_dict_item(self.cfg.path_data, split)
        if cfg_from_yaml:
            self.cfg.NUM = len(self)
        
        self.collate_ver = self.cfg.get('collate_fn', 'v2_0') # Post-processing
    
    def load_dict_item(self, path_data, split):
        def get_split(split_txt, list_dict_split, val):
            f = open(split_txt, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                seq, label = line.split(',')
                list_dict_split[int(seq)][label.rstrip('\n')] = val
        list_dict_split = [dict() for _ in range(58+1)]
        get_split(path_data.split[0], list_dict_split, 'train')
        get_split(path_data.split[1], list_dict_split, 'test')
        
        list_seqs_w_header = []
        for path_header in path_data.list_dir_kradar:
            list_seqs = os.listdir(path_header)
            list_seqs_w_header.extend([(seq, path_header) for seq in list_seqs])
        list_seqs_w_header = sorted(list_seqs_w_header, key=lambda x: int(x[0]))

        list_dict_item = []
        dict_seq_changed_idx = dict()
        for idx_seq in range(58):
            temp_key = f'{idx_seq+1}'
            dict_seq_changed_idx[temp_key] = []
        list_check_duplicated_seq = []
        prev_seq = None
        idx_iter = 0
        for seq, path_header in list_seqs_w_header:
            list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
            if seq not in list_check_duplicated_seq: # new seq
                now_split = 'train'
                if len(list_check_duplicated_seq) != 0:
                    dict_seq_changed_idx[prev_seq].append(idx_iter)
                list_check_duplicated_seq.append(seq)
                dict_seq_changed_idx[seq].append(idx_iter)
                prev_seq = seq
            for label in list_labels:
                path_label_v1_0 = osp.join(path_header, seq, 'info_label', label)
                path_label_v1_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v1_1/{seq}_info_label_revised', label)
                path_label_v2_0 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_0', 'KRadar_refined_label_by_UWIPL', seq, label)
                path_label_v2_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_1', 'KRadar_revised_visibility', seq, label)
                temp_split = list_dict_split[int(seq)][label]
                dict_item = dict(
                    meta = dict(
                        header = path_header, seq = seq,
                        label_v1_0 = path_label_v1_0, label_v1_1 = path_label_v1_1,
                        label_v2_0 = path_label_v2_0, label_v2_1 = path_label_v2_1,
                        split = temp_split,
                    ),
                )
                if temp_split != now_split:
                    dict_seq_changed_idx[seq].append(idx_iter)
                    if now_split == 'train':
                        now_split = 'test'
                    elif now_split == 'test':
                        now_split = 'train'
                if self.load_label_in_advance:
                    dict_item = self.get_label(dict_item)
                list_dict_item.append(dict_item)
                idx_iter += 1
        dict_seq_changed_idx[prev_seq].append(idx_iter)
        
        # Joint dicts
        list_joint_dict_item = []
        joint_samples = self.joint_samples
        for idx_seq in range(58):
            temp_key = f'{idx_seq+1}'
            idx_0, idx_1, idx_2, idx_3, idx_4 = dict_seq_changed_idx[temp_key]
            part_0 = list_dict_item[idx_0:idx_1] # train
            part_1 = list_dict_item[idx_1:idx_2] # test
            part_2 = list_dict_item[idx_2:idx_3] # train
            part_3 = list_dict_item[idx_3:idx_4] # test

            if split == 'all' or split == 'train':
                for idx_dict in range(len(part_0)-joint_samples+1):
                    part_0[idx_dict]['following'] = part_0[idx_dict+1:idx_dict+joint_samples]
                    list_joint_dict_item.append(part_0[idx_dict])
                for idx_dict in range(len(part_2)-joint_samples+1):
                    part_2[idx_dict]['following'] = part_2[idx_dict+1:idx_dict+joint_samples]
                    list_joint_dict_item.append(part_2[idx_dict])
            
            if split == 'all' or split == 'test':
                for idx_dict in range(len(part_1)-joint_samples+1):
                    part_1[idx_dict]['following'] = part_1[idx_dict+1:idx_dict+joint_samples]
                    list_joint_dict_item.append(part_1[idx_dict])
                for idx_dict in range(len(part_3)-joint_samples+1):
                    part_3[idx_dict]['following'] = part_3[idx_dict+1:idx_dict+joint_samples]
                    list_joint_dict_item.append(part_3[idx_dict])

        # Filter unavailable frames (frames wo objects) (only)
        if self.label.remove_0_obj:
            list_joint_dict_item = list(filter(lambda item: item['meta']['num_obj']>0, list_joint_dict_item))

        return list_joint_dict_item
    
    def get_label(self, dict_item):
        meta = dict_item['meta']
        temp_key = 'label_' + self.label_version
        path_label = meta[temp_key]
        ver = self.label_version

        f = open(path_label)
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
            # print('* b4: ', lines)
            lines.insert(1, '*'+line0)
            lines[0] = header
            # print('* after: ', lines)
        rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
        tstamp = tstamp.split('=')[1]
        dict_idx = dict(rdr=rdr, ldr64=ldr64, camf=camf,\
                        ldr128=ldr128, camr=camr, tstamp=tstamp)
        if ver == 'v1_0':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                if len(list_vals) != 11:
                    print('* split err in ', path_label)
                    continue
                idx_p = int(list_vals[1])
                idx_b4 = int(list_vals[2])
                cls_name = list_vals[3]
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p, idx_b4), 'R'))
        elif ver == 'v1_1':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(',')
                if len(list_vals) != 12:
                    print('* split err in ', path_label)
                    continue
                avail = (list_vals[1]).lstrip(' ').rstrip(' ')
                try:
                    idx_p = int(list_vals[2])
                    idx_b4 = int(list_vals[3])
                except:
                    print('* split err in ', path_label)
                    continue
                cls_name = (list_vals[4]).lstrip(' ').rstrip(' ')
                x = float(list_vals[5])
                y = float(list_vals[6])
                z = float(list_vals[7])
                th = float(list_vals[8])*deg2rad
                l = 2*float(list_vals[9])
                w = 2*float(list_vals[10])
                h = 2*float(list_vals[11])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p, idx_b4), avail))
        elif ver == 'v2_0':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                idx_p = int(list_vals[1])
                idx_p = dict_item['meta']['seq'] + '_' + f'{idx_p}'
                cls_name = (list_vals[2])
                x = float(list_vals[3])
                y = float(list_vals[4])
                z = float(list_vals[5])
                th = float(list_vals[6])*deg2rad
                l = 2*float(list_vals[7])
                w = 2*float(list_vals[8])
                h = 2*float(list_vals[9])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), 'R'))
        elif ver == 'v2_1':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                avail = list_vals[1]
                idx_p = int(list_vals[2])
                cls_name = (list_vals[3])
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), avail))

        header = dict_item['meta']['header']
        seq = dict_item['meta']['seq']
        path_calib = osp.join(header, seq, 'info_calib', 'calib_radar_lidar.txt')
        dict_path = dict(
            calib = path_calib,
            ldr64 = osp.join(header, seq, 'os2-64', f'os2-64_{ldr64}.pcd'),
            desc = osp.join(header, seq, 'description.txt'),
        )

        onlyR = self.label.onlyR
        consider_cls = self.label.consider_cls
        if consider_cls | onlyR:
            list_temp = []
            for obj in list_tuple_objs:
                cls_name, _, _, avail = obj
                if consider_cls:
                    is_consider, _, _, _ = self.label[cls_name]
                    if not is_consider:
                        continue
                if onlyR:
                    if avail != 'R':
                        continue
                list_temp.append(obj)
            list_tuple_objs = list_temp

        dict_item['meta']['calib'] = self.get_calib_values(path_calib) if self.item.calib else None
        if self.label.calib:
            list_temp = []
            dx, dy, dz = dict_item['meta']['calib']
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                x = x + dx
                y = y + dy
                z = z + dz
                list_temp.append((cls_name, (x, y, z, th, l, w, h), trk, avail))
            list_tuple_objs = list_temp

        if self.label.consider_roi: # after calib
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            check_azimuth_for_rdr = self.roi.check_azimuth_for_rdr
            azimuth_min, azimuth_max = self.roi.azimuth_deg
            rad2deg = 180./np.pi
            temp_list = []
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                azimuth = np.arctan2(y, x)*rad2deg
                if check_azimuth_for_rdr & ((azimuth < azimuth_min) | (azimuth > azimuth_max)):
                    continue
                if (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max) | (z < z_min) | (z > z_max):
                    continue
                temp_list.append(obj)
            list_tuple_objs = temp_list

        num_obj = len(list_tuple_objs)

        dict_item['meta'].update(dict(
            path=dict_path, idx=dict_idx, label=list_tuple_objs, num_obj=num_obj))
        return dict_item
    
    def get_calib_values(self, path_calib):
        f = open(path_calib, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        list_values = [list_calib[1], list_calib[2], self.calib['z_offset']] # X, Y, Z
        return list_values
    
    def get_description(self, dict_item): # ./tools/tag_generator
        f = open(dict_item['meta']['path']['desc'])
        line = f.readline()
        road_type, capture_time, climate = line.split(',')
        dict_desc = {
            'capture_time': capture_time,
            'road_type': road_type,
            'climate': climate,
        }
        f.close()
        dict_item['meta']['desc'] = dict_desc
        
        return dict_item
    
    def get_ldr64(self, dict_item):
        if self.ldr64.processed: # with attr & calib & roi
            pass # TODO
        else:
            with open(dict_item['meta']['path']['ldr64'], 'r') as f:
                lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
                pc_lidar = [point.split() for point in lines]
                f.close()
            pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

            if self.ldr64.inside_ldr64:
                pc_lidar = pc_lidar[np.where(
                    (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                    (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]
            
            if self.ldr64.calib:
                n_pts, _ = pc_lidar.shape
                calib_vals = np.array(dict_item['meta']['calib']).reshape(-1,3).repeat(n_pts, axis=0)
                pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        dict_item['ldr64'] = pc_lidar

        return dict_item
    
    def get_ldr64_from_path(self, path_ldr64):
        with open(path_ldr64, 'r') as f:
            lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

        if self.ldr64.inside_ldr64:
            pc_lidar = pc_lidar[np.where(
                (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]
        
        if self.ldr64.calib:
            n_pts, _ = pc_lidar.shape
            calib_vals = np.array([-2.54,0.3,0.7]).reshape(-1,3).repeat(n_pts, axis=0)
            pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        return pc_lidar
    
    def get_rdr_sparse(self, dict_item):
        if self.rdr_sparse.processed:
            dir_rdr_sparse = self.rdr_sparse.dir
            seq = dict_item['meta']['seq']
            rdr_idx = dict_item['meta']['idx']['rdr']
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'sprdr_{rdr_idx}.npy')
            rdr_sparse = np.load(path_rdr_sparse)
        else: # from cube or tesseract (TODO)
            pass
        dict_item['rdr_sparse'] = rdr_sparse
        
        return dict_item
    
    def filter_roi(self, dict_item):
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
        list_keys = self.roi.keys
        for temp_key in list_keys:
            if temp_key in ['rdr_sparse', 'ldr64']:
                temp_data = dict_item[temp_key]
                temp_data = temp_data[np.where(
                    (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                    (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                    (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
                dict_item[temp_key] = temp_data
            # elif temp_key == 'label': # moved to dict item
        
        return dict_item

    def __len__(self):
        return len(self.list_dict_item)
    
    def __getitem__(self, idx):
        dict_item = self.list_dict_item[idx]
        dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item
        dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
        dict_item = self.get_rdr_sparse(dict_item) if self.item['rdr_sparse'] else dict_item
        dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item
        dict_item = self.get_description(dict_item)

        return dict_item
    
    def getitem_with_dict_item(self, dict_item):
        dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item
        dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
        dict_item = self.get_rdr_sparse(dict_item) if self.item['rdr_sparse'] else dict_item
        dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item
        dict_item = self.get_description(dict_item)

        return dict_item

    ### Vis ###
    def create_cylinder_mesh(self, radius, p0, p1, color=[1, 0, 0]):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(np.array(p1)-np.array(p0)))
        cylinder.paint_uniform_color(color)
        frame = np.array(p1) - np.array(p0)
        frame /= np.linalg.norm(frame)
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(frame[2]), np.arctan2(-frame[0], frame[1]), 0))
        cylinder.rotate(R, center=[0, 0, 0])
        cylinder.translate((np.array(p0) + np.array(p1)) / 2)
        return cylinder
    
    def draw_3d_box_in_cylinder(self, vis, center, theta, l, w, h, color=[1, 0, 0], radius=0.1, in_cylinder=True):
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
                cylinder = self.create_cylinder_mesh(radius, corners_rotated[line[0]], corners_rotated[line[1]], color)
                vis.add_geometry(cylinder)
        else:
            vis.add_geometry(line_set)

    def create_sphere(self, radius=0.2, resolution=30, rgb=[0., 0., 0.], center=[0., 0., 0.]):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        color = np.array(rgb)
        mesh_sphere.vertex_colors = o3d.utility.Vector3dVector([color for _ in range(len(mesh_sphere.vertices))])
        x, y, z = center
        transform = np.identity(4)
        transform[0, 3] = x
        transform[1, 3] = y
        transform[2, 3] = z
        mesh_sphere.transform(transform)
        return mesh_sphere
    
    def vis_in_open3d(self, dict_item, vis_list=['rdr_sparse', 'ldr64', 'label']):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        if 'ldr64' in vis_list:
            pc_lidar = dict_item['ldr64']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
            vis.add_geometry(pcd)

        if 'rdr_sparse' in vis_list:
            rdr_sparse = dict_item['rdr_sparse']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)
        
        if 'label' in vis_list:
            label = dict_item['meta']['label']
            for obj in label:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                consider, logit_idx, rgb, bgr = self.label[cls_name]
                if consider:
                    self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
            vis.run()
            vis.destroy_window()
    ### Vis ###

    ### Distribution ###
    def get_distribution_of_label(self, consider_avail=True):
        dict_label = self.label.copy()
        dict_label.pop('calib')
        dict_label.pop('onlyR')
        dict_label.pop('Label')
        dict_label.pop('consider_cls')
        dict_label.pop('consider_roi')
        dict_label.pop('remove_0_obj')
        
        dict_for_dist = dict()
        dict_for_value = dict()
        for obj_name in dict_label.keys():
            dict_for_dist[obj_name] = 0
            dict_for_value[obj_name] = [0., 0., 0.]
        
        if consider_avail:
            dict_avail = dict()
            list_avails = ['R', 'L', 'L1']
            for avail in list_avails:
                dict_temp = dict()
                for obj_name in dict_label.keys():
                    dict_temp[obj_name] = 0
                dict_avail[avail] = dict_temp

        for dict_item in tqdm(self.list_dict_item):
            dict_item = self.get_label(dict_item)
            for obj in dict_item['meta']['label']:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                dict_for_dist[cls_name] += 1
                dict_for_value[cls_name][0] += l
                dict_for_value[cls_name][1] += w
                dict_for_value[cls_name][2] += h
                try:
                    if consider_avail:
                        dict_avail[avail][cls_name] += 1
                except:
                    print(dict_item['meta']['label_v2_1'])

        for obj_name in dict_for_dist.keys():
            n_obj = dict_for_dist[obj_name]
            l, w, h = dict_for_value[obj_name]
            print('* # of ', obj_name, ': ', n_obj)
            divider = np.maximum(n_obj, 1)
            print('* lwh of ', obj_name, ': ', l/divider, ', ', w/divider, ', ', h/divider)
        
        if consider_avail:
            for avail in list_avails:
                print('-'*30, avail, '-'*30)
                for obj_name in dict_avail[avail].keys():
                    print('* # of ', obj_name, ': ', dict_avail[avail][obj_name])
    ### Distribution ###

    def collate_fn(self, list_batch):
        if None in list_batch:
            print('* Exception error (Dataset): collate fn 0')
            return None
        
        if self.joint_samples is not None:
            list_new_batch = []
            for dict_item in list_batch:
                list_new_batch.append(dict_item)
                for temp_dict_item in dict_item['following']:
                    list_new_batch.append(self.getitem_with_dict_item(temp_dict_item))
            list_batch = list_new_batch

        if self.collate_ver == 'v1_0':
            dict_batch = dict()
            
            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []

            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []

                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                    elif k in ['rdr_sparse', 'ldr64']:
                        dict_batch[k].append(torch.from_numpy(dict_item[k]).float())
            dict_batch['batch_size'] = batch_idx+1

            for k in list_keys:
                if k in ['rdr_sparse', 'ldr64']:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))
                    
                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)
        
        elif self.collate_ver == 'v2_0': # gt_boxes (B, M, 8)
            dict_batch = dict()
            
            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []
            dict_batch['gt_boxes'] = []

            max_objs = 0 # for gt_boxes (M)
            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []
                        list_gt_boxes = []
                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                            x, y, z, th, l, w, h = vals
                            list_gt_boxes.append([x, y, z, l, w, h, th, logit_idx])
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                        dict_batch['gt_boxes'].append(list_gt_boxes)
                        max_objs = max(max_objs, dict_item['meta']['num_obj'])
                    elif k in ['rdr_sparse', 'ldr64']:
                        dict_batch[k].append(torch.from_numpy(dict_item[k]).float())
            dict_batch['batch_size'] = batch_idx+1

            batch_size = dict_batch['batch_size']
            gt_boxes = np.zeros((batch_size, max_objs, 8))
            for batch_idx in range(batch_size):
                gt_box = np.array(dict_batch['gt_boxes'][batch_idx])
                gt_boxes[batch_idx,:dict_batch['num_objs'][batch_idx],:] = gt_box
            dict_batch['gt_boxes'] = torch.tensor(gt_boxes, dtype=torch.float32)

            for k in list_keys:
                if k in ['rdr_sparse', 'ldr64']:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))
                    
                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)

        # print(dict_batch['batch_size'])

        return dict_batch

if __name__ == '__main__':
    kradar_detection = KRadarTracking_v1_0(split='all')
    print(len(kradar_detection)) # 34994 for all

    # dict_item = kradar_detection[5000]
    # kradar_detection.vis_in_open3d(dict_item, ['ldr64', 'label', 'rdr_sparse'])
    kradar_detection.get_distribution_of_label()
