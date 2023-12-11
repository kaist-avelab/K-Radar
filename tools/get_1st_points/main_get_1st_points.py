'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import numpy as np
import open3d as o3d

from scipy.io import loadmat # from matlab
from tqdm import tqdm
from easydict import EasyDict

roi = [0,-15,-2,72,15,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/ave/HDD_3_1/radar_bin_lidar_bag_files/generated_files',\
                           '/media/ave/HDD_3_1/gen_2to5',\
                           '/media/ave/HDD_3_2/radar_bin_lidar_bag_files/generated_files',\
                           '/media/ave/data_2/radar_bin_lidar_bag_files/generated_files'],
        split = ['./resources/split/train.txt', './resources/split/test.txt'],
        revised_label_v1_1 = './tools/revise_label/kradar_revised_label_v1_1',
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
        revised_label_v2_1 = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility',
    ),
    label = { # (consider, logit_idx, rgb, bgr)
        'calib':            True,
        'onlyR':            False,
        'consider_cls':     False,
        'consider_roi':     True,
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
        dir='/media/ave/sub4/rtnh_wider_1p_1'
    ),
    roi = dict(filter=True, xyz=roi, keys=['ldr64', 'rdr_sparse'], check_azimuth_for_rdr=True, azimuth_deg=[-53,53]),
)

class Get1stPoints():
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

        self.list_dict_item = self.load_dict_item(self.cfg.path_data, split)
        if cfg_from_yaml:
            self.cfg.NUM = len(self)
        
        self.collate_ver = self.cfg.get('collate_fn', 'v1_0') # Post-processing

        self.arr_range, self.arr_azimuth, self.arr_elevation, \
            self.arr_doppler = self.load_physical_values(is_with_doppler=True)
    
        arr_r = self.arr_range
        arr_a = self.arr_azimuth
        arr_e = self.arr_elevation

        r_min = np.min(arr_r)
        r_bin = np.mean(arr_r[1:]-arr_r[:-1])
        r_max = np.max(arr_r)
        
        a_min = np.min(arr_a)
        a_bin = np.mean(arr_a[1:]-arr_a[:-1])
        a_max = np.max(arr_a)

        e_min = np.min(arr_e)
        e_bin = np.mean(arr_e[1:]-arr_e[:-1])
        e_max = np.max(arr_e)

        self.info_rae = [
            [r_min, r_bin, r_max],
            [a_min, a_bin, a_max],
            [e_min, e_bin, e_max]]


    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat('./resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

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
                if self.load_label_in_advance:
                    dict_item = self.get_label(dict_item)
                list_dict_item.append(dict_item)

        if split == 'all':
            pass
        else:
            list_dict_item = list(filter(lambda item: item['meta']['split']==split, list_dict_item))

        # Filter unavailable frames (frames wo objects) (only)
        if self.label.remove_0_obj:
            list_dict_item = list(filter(lambda item: item['meta']['num_obj']>0, list_dict_item))

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
        if ver == 'v2_0':
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

        if 'rdr_pc' in vis_list:
            rdr_pc = dict_item['rdr_pc']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_pc[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)

        if 'ldr64_quantized' in vis_list:
            ldr64_quantized = dict_item['ldr64_quantized']
            pcd_ldr = o3d.geometry.PointCloud()
            pcd_ldr.points = o3d.utility.Vector3dVector(ldr64_quantized[:,:3])
            pcd_ldr.paint_uniform_color([1.,0.,1.])
            vis.add_geometry(pcd_ldr)

        if 'pc_from_quantized_indices' in vis_list:
            ldr64_quantized = dict_item['pc_from_quantized_indices']
            pcd_ldr = o3d.geometry.PointCloud()
            pcd_ldr.points = o3d.utility.Vector3dVector(ldr64_quantized[:,:3])
            pcd_ldr.paint_uniform_color([0.,1.,1.])
            vis.add_geometry(pcd_ldr)
        
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

    ### Get 1st points ###
    def get_tesseract(self, dict_item):
        seq = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr']
        path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')            
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2)) # DRAE
        dict_item['tesseract'] = arr_tesseract

        return dict_item
    
    def get_cube_polar(self, dict_item):
        dict_item = self.get_tesseract(dict_item)
        tesseract = dict_item['tesseract'][1:,:,:,:]/(1e+13)
        cube_pw = np.mean(tesseract, axis=0, keepdims=False)

        # (1) softmax (not used: overflow)    
        # tesseract_exp = np.exp(tesseract)
        # tesseract_exp_sum = np.repeat(np.sum(tesseract_exp, axis=0, keepdims=True), 63, axis=0)
        # tesseract_dist = tesseract_exp/tesseract_exp_sum

        # (2) sum
        tesseract_sum = np.repeat(np.sum(tesseract, axis=0, keepdims=True), 63, axis=0)
        tesseract_dist = tesseract/tesseract_sum

        tesseract_dop = np.reshape(self.arr_doppler[1:], (63,1,1,1)).repeat(256,1).repeat(107,2).repeat(37,3)
        cube_dop = np.sum(tesseract_dist*tesseract_dop, axis=0, keepdims=False)
        
        dict_item['cube_pw_polar'] = cube_pw
        dict_item['cube_dop_cartesian'] = cube_dop

        return dict_item
    
    def get_portional_rdr_points_from_tesseract(self, dict_item, rate=0.001):
        dict_item = self.get_cube_polar(dict_item)

        cube_pw = dict_item['cube_pw_polar'] # Normalized with 1e+13
        cube_dop = dict_item['cube_dop_cartesian'] # Expectation w/ pw dist

        extracted_ind = np.where(cube_pw > np.quantile(cube_pw, 1-rate))
        r_ind, a_ind, e_ind = extracted_ind
        pw = cube_pw[extracted_ind]
        dop = cube_dop[extracted_ind]
        
        r = self.arr_range[r_ind]
        az = self.arr_azimuth[a_ind]
        el = self.arr_elevation[e_ind]

        # Radar polar to General polar coordinate
        az = -az
        el = -el

        # For flipped azimuth & elevation angle
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        dict_item['rdr_pc'] = np.stack((x,y,z,pw,dop), axis=1)
        
        return dict_item
    
    def get_1st_points_from_lpc_for_4drt(self, dict_item):
        dict_item = self.get_ldr64(dict_item)
        pc_lidar = dict_item['ldr64']
        pc_lidar_front = pc_lidar[np.where(pc_lidar[:,0]>0.)[0],:]

        r_min, r_bin, r_max = self.info_rae[0]
        a_min, a_bin, a_max = self.info_rae[1]
        e_min, e_bin, e_max = self.info_rae[2]

        n_r = int(np.around((r_max-r_min)/r_bin))+1
        n_a = int(np.around((a_max-a_min)/a_bin))+1
        n_e = int(np.around((e_max-e_min)/e_bin))+1

        arr_r = np.linspace(r_min, r_max, n_r)
        arr_a = np.linspace(a_min, a_max, n_a)
        arr_e = np.linspace(e_min, e_max, n_e)

        # print(arr_r, arr_a, arr_e)
        # print(n_r, n_a, n_e)

        x = pc_lidar_front[:,0:1]
        y = pc_lidar_front[:,1:2]
        z = pc_lidar_front[:,2:3]

        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.arctan2(y, x)
        el = np.arcsin(z / r)

        rae_list = np.concatenate((r, az, el), axis=1) # filter with min/max
        rae_list = rae_list[np.where(
            (r>=r_min) & (r<r_max) & (az>=a_min) & (az<a_max) & (el>=e_min) & (el<e_max))[0],:]
        r_indices = (np.round((rae_list[:,0]-r_min)/r_bin)).astype(int)
        a_indices = (np.round((rae_list[:,1]-a_min)/a_bin)).astype(int)
        e_indices = (np.round((rae_list[:,2]-e_min)/e_bin)).astype(int)

        rae_tensor = np.full((n_r, n_a, n_e), False, dtype=bool) # rae
        rae_tensor[r_indices, a_indices, e_indices] = True

        # Sampling 1st points in range
        rae_tensor_temp = np.full_like(rae_tensor, False, dtype=bool)
        for a in range(n_a):
            for e in range(n_e):
                range_temp = rae_tensor[:,a,e]
                ind_r_temp = np.where(range_temp == True)[0]
                if len(ind_r_temp) == 0:
                    continue
                min_ind = np.min(ind_r_temp)
                rae_tensor_temp[min_ind,a,e] = True
        rae_tensor = rae_tensor_temp

        r_indices, a_indices, e_indices = np.where(rae_tensor == True)
        r = (arr_r[r_indices]).reshape(-1,1)
        az = (arr_a[a_indices]).reshape(-1,1)
        el = (arr_e[e_indices]).reshape(-1,1)

        # Polar to Cartesian
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        dict_item['ldr64_quantized'] = np.concatenate((x,y,z), axis=1)

        # Global polar to Radar polar coordinate
        # +1 for matlab representation
        rdr_ind_r = r_indices+1
        rdr_ind_a = (n_a-a_indices-1)+1 # 0 -> 106 -> 107
        rdr_ind_e = (n_e-e_indices-1)+1 # 0 -> 36 -> 37

        rdr_ind_r = rdr_ind_r.reshape(-1,1)
        rdr_ind_a = rdr_ind_a.reshape(-1,1)
        rdr_ind_e = rdr_ind_e.reshape(-1,1)

        dict_item['ldr64_quantized_indices'] = np.concatenate((rdr_ind_r, rdr_ind_a, rdr_ind_e), axis=1)

        return dict_item
    
    def get_points_from_quantized_indices(self, dict_item):
        quantized_indices = dict_item['ldr64_quantized_indices']
        rdr_ind_r = quantized_indices[:,0]-1
        rdr_ind_a = quantized_indices[:,1]-1
        rdr_ind_e = quantized_indices[:,2]-1 # -1 for python representation
        
        r = self.arr_range[rdr_ind_r]
        az = self.arr_azimuth[rdr_ind_a]
        el = self.arr_elevation[rdr_ind_e]

        # Radar polar to General polar coordinate
        az = -az
        el = -el

        # Polar to Cartesian
        x = r * np.cos(el) * np.cos(az) + 0.01 # +0.01 is not to be overlapped
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        dict_item['pc_from_quantized_indices'] = np.stack((x,y,z), axis=1)

    def save_quantized_indices(self, root_path='/media/ave/data_2/kradar_quantized_indices_matlab'):
        for seq_name in range(58):
            seq_folder = osp.join(root_path, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)
        
        for idx_sample in tqdm(range(len(self))):
            dict_item = self.__getitem__(idx_sample)
            self.get_ldr64(dict_item)
            self.get_1st_points_from_lpc_for_4drt(dict_item)
            
            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            rdr_idx = dict_meta['idx']['rdr']
            ldr64_quantized_indices = dict_item['ldr64_quantized_indices']

            path_quantized_indices = osp.join(root_path, seq_name, f'ind_{rdr_idx}.npy')
            np.save(path_quantized_indices, ldr64_quantized_indices)

    def visaulize_1st_points_from_lpc(self, dict_item):
        dict_item = self.get_ldr64(dict_item)
        pc_lidar = dict_item['ldr64']
        pc_lidar_front = pc_lidar[np.where(pc_lidar[:,0]>0.)[0],:]

        ### Hyper-params ###
        deg2rad = np.pi/180. # [m, deg, deg]
        r_tensor_size = np.array([0., 0.2, 100.])
        az_tensor_size = np.array([-60., 1., 60.])*deg2rad
        el_tensor_size = np.array([-20., 1., 20.])*deg2rad

        r_min, r_bin, r_max = r_tensor_size
        a_min, a_bin, a_max = az_tensor_size
        e_min, e_bin, e_max = el_tensor_size

        n_r = int(np.around((r_max-r_min)/r_bin))
        n_a = int(np.around((a_max-a_min)/a_bin))
        n_e = int(np.around((e_max-e_min)/e_bin))

        arr_r = np.linspace(r_min, r_max-r_bin, n_r) + r_bin/2.
        arr_a = np.linspace(a_min, a_max-a_bin, n_a) + a_bin/2.
        arr_e = np.linspace(e_min, e_max-e_bin, n_e) + e_bin/2.
        ### Hyper-params ###

        ### Cartesian to Polar ###
        x = pc_lidar_front[:,0:1]
        y = pc_lidar_front[:,1:2]
        z = pc_lidar_front[:,2:3]

        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.arctan2(y, x)
        el = np.arcsin(z / r)
        ### Cartesian to Polar ###

        ### (1) Filter with min/max & Quantization ###
        rae_list = np.concatenate((r, az, el), axis=1) # filter with min/max
        rae_list = rae_list[np.where(
            (r>=r_min) & (r<r_max) & (az>=a_min) & (az<a_max) & (el>=e_min) & (el<e_max))[0],:]
        # print(rae_list.shape)
        r_indices = ((rae_list[:,0]-r_min)/r_bin).astype(int)
        a_indices = ((rae_list[:,1]-a_min)/a_bin).astype(int)
        e_indices = ((rae_list[:,2]-e_min)/e_bin).astype(int)
        # print(r_indices.shape)
        ### (1) Filter with min/max & Quantization ###

        ### Polar to Polar occupied tensor ###
        rae_tensor = np.full((n_r, n_a, n_e), False, dtype=bool) # rae
        rae_tensor[r_indices, a_indices, e_indices] = True
        ### Polar to Polar ocuupied tensor ###

        ### (2) Sampling 1st points in range ###
        rae_tensor_temp = np.full_like(rae_tensor, False, dtype=bool)
        for a in range(n_a):
            for e in range(n_e):
                range_temp = rae_tensor[:,a,e]
                ind_r_temp = np.where(range_temp == True)[0]
                if len(ind_r_temp) == 0:
                    continue
                min_ind = np.min(ind_r_temp)
                rae_tensor_temp[min_ind,a,e] = True
        rae_tensor = rae_tensor_temp
        ### (2) Sampling 1st points in range ###

        ### (3) RAE tensor to r, az, el values ###
        r_indices, a_indices, e_indices = np.where(rae_tensor == True)
        r = (arr_r[r_indices]).reshape(-1,1)
        az = (arr_a[a_indices]).reshape(-1,1)
        el = (arr_e[e_indices]).reshape(-1,1)
        ### (3) RAE tensor to r, az, el values ###

        ### Polar to Cartesian ###
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        ### Polar to Cartesian ###

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate((x,y,z), axis=1))
        # pcd.points = o3d.utility.Vector3dVector(pc_lidar_front[:,:3])
        vis.add_geometry(pcd)

        label = dict_item['meta']['label']
        for obj in label:
            cls_name, (x, y, z, th, l, w, h), trk, avail = obj
            consider, logit_idx, rgb, bgr = self.label[cls_name]
            if consider:
                self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
        
        vis.run()
        vis.destroy_window()
    ### Get 1st points ###

if __name__ == '__main__':
    get_1st_points = Get1stPoints(split='all')
    print('* Total samples: ', len(get_1st_points)) # 10764

    ### Visualization ###
    # dict_item = get_1st_points[3000]
    # # get_1st_points.vis_in_open3d(dict_item, ['ldr64', 'label', 'rdr_sparse'])
    # # get_1st_points.visaulize_1st_points_from_lpc(dict_item)

    # # get_1st_points.get_tesseract(dict_item)
    # # get_1st_points.get_cube_polar(dict_item)
    # get_1st_points.get_ldr64(dict_item)
    # get_1st_points.get_portional_rdr_points_from_tesseract(dict_item, rate=0.001)
    # get_1st_points.get_1st_points_from_lpc_for_4drt(dict_item)
    # get_1st_points.get_points_from_quantized_indices(dict_item)

    # get_1st_points.vis_in_open3d(dict_item, ['label', 'pc_from_quantized_indices'])
    ### Visualization ###

    ### Generate samples ###
    get_1st_points.save_quantized_indices()
    ### Generate samples ###

