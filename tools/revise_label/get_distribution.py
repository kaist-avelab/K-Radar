'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, Dongin Kim, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, dongin.kim@kaist.ac.kr
* description: revise label 2.1 -> 2.2 for 3D object detection
'''

import os
import os.path as osp
import numpy as np
import open3d as o3d

from tqdm import tqdm
from easydict import EasyDict

dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/ave/HDD_4_1/gen_2to5', '/media/ave/HDD_4_1/radar_bin_lidar_bag_files/generated_files', '/media/ave/e95e0722-32a4-4880-a5d5-bb46967357d6/radar_bin_lidar_bag_files/generated_files', '/media/ave/4f089d0e-7b60-493d-aac7-86ead9655194/radar_bin_lidar_bag_files/generated_files'],
        split = ['./resources/split/train.txt', './resources/split/test.txt'],
        revised_label_v1_1 = './tools/revise_label/kradar_revised_label_v1_1',
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
        revised_label_v2_1 = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility',
    ),
    label_version = 'v2_1', # ['v1_0', 'v1_1', 'v2_0', v2_1']
    item = dict(calib=False, ldr64=True, ldr128=False, rdr=True, rdr_sparse=False, cam=False),
    calib = dict(z_offset=0.7),
    ldr64 = dict(processed=False, skip_line=13, n_attr=9, inside_ldr64=True, calib=False,),
    rdr = dict(cube=False,),
    rdr_sparse = dict(),
    roi = dict(filter=True, xyz=[0,72,-15,15,-2,6]),
    label = { # (consider, rgb, bgr)
        'calib':            False,
        'onlyR':            True,
        'Sedan':            (True,  [0, 1, 0],       [0,255,0]),
        'Bus or Truck':     (True,  [1, 0.2, 0],     [0,50,255]),
        'Motorcycle':       (True,  [1, 0, 0],       [0,0,255]),
        'Bicycle':          (True,  [1, 1, 0],       [0,255,255]),
        'Bicycle Group':    (True,  [0, 0.5, 1],     [0,128,255]),
        'Pedestrian':       (True,  [0, 0, 1],       [255,0,0]),
        'Pedestrian Group': (True,  [0.4, 0, 1],     [255,0,100]),
        'Label':            (False, [0.5, 0.5, 0.5], [128,128,128]),
    },
)

class KRadarDetection_v2_0(object):
    def __init__(self, cfg=None):
        if cfg == None:
            cfg = EasyDict(dict_cfg)
        self.cfg = cfg
        self.list_dict_item = self.load_dict_item(cfg.path_data)
        self.label_version = self.cfg.get('label_version', 'v1_0')
        
        self.item = self.cfg.item
        self.calib = self.cfg.calib
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.roi = self.cfg.roi

        self.label = self.cfg.label
    
    def load_dict_item(self, path_data):
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
        for seq, path_header in list_seqs_w_header:
            list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
            for label in list_labels:
                path_label_v1_0 = osp.join(path_header, seq, 'info_label', label)
                path_label_v1_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v1_1/{seq}_info_label_revised', label)
                path_label_v2_0 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_0', 'KRadar_refined_label_by_UWIPL', seq, label)
                path_label_v2_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_1', 'KRadar_revised_visibility', seq, label)
                dict_item = dict(
                    meta = dict(
                        header = path_header, seq = seq,
                        label_v1_0 = path_label_v1_0, label_v1_1 = path_label_v1_1,
                        label_v2_0 = path_label_v2_0, label_v2_1 = path_label_v2_1,
                        split = list_dict_split[int(seq)][label]
                    ),
                )
                list_dict_item.append(dict_item)
        return list_dict_item
    
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
        path_calib = osp.join(header, seq, 'calib_radar_lidar.txt')
        dict_path = dict(
            calib = path_calib,
            ldr64 = osp.join(header, seq, 'os2-64', f'os2-64_{ldr64}.pcd')
        )

        dict_item['meta']['calib'] = self.get_calib_values(path_calib) if self.item.calib else None
        if self.label.calib:
            list_temp = []
            dx, dy, dz = dict_item['meta']['calib']
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                x = x + dx
                y = y + dy
                z = z + dz
                list_temp.append(cls_name, (x, y, z, th, l, w, h), trk, avail)

        dict_item['meta'].update(dict(
            path = dict_path, idx = dict_idx, label=list_tuple_objs))
        return dict_item
    
    def get_calib_values(self, path_calib):
        f = open(path_calib, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        list_values = [list_calib[1], list_calib[2], self.calib['z_offset']] # X, Y, Z
        return list_values
    
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

    def __len__(self):
        return len(self.list_dict_item)
    
    def __getitem__(self, idx):
        dict_item = self.list_dict_item[idx]
        dict_item = self.get_label(dict_item)
        dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item

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
        
        if 'label' in vis_list:
            label = dict_item['meta']['label']
            for obj in label:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                consider, rgb, bgr = self.label[cls_name]
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
        
        dict_for_dist = dict()
        for obj_name in dict_label.keys():
            dict_for_dist[obj_name] = 0
        
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
                try:
                    if consider_avail:
                        dict_avail[avail][cls_name] += 1
                except:
                    print(dict_item['meta']['label_v2_1'])

        for obj_name in dict_for_dist.keys():
            print('* # of ', obj_name, ': ', dict_for_dist[obj_name])
        
        if consider_avail:
            for avail in list_avails:
                print('-'*30, avail, '-'*30)
                for obj_name in dict_avail[avail].keys():
                    print('* # of ', obj_name, ': ', dict_avail[avail][obj_name])
    ### Distribution ###

if __name__ == '__main__':
    kradar_detection = KRadarDetection_v2_0()
    # print(len(kradar_detection)) # 34994

    # dict_item = kradar_detection[0]
    # kradar_detection.vis_in_open3d(dict_item, ['ldr64', 'label'])
    kradar_detection.get_distribution_of_label()
