import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d
import cv2
import yaml

from tqdm import tqdm
from easydict import EasyDict

from scipy.io import loadmat # from matlab

roi = [0,-15,-2,72,15,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/donghee/kradar/dataset'],
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
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
    item = dict(calib=True, ldr64=True, rdr_sparse=True, rdr_polar_3d=True),
    calib = dict(z_offset=0.7),
    ldr64 = dict(processed=False, skip_line=13, n_attr=9, inside_ldr64=True, calib=True,),
    rdr_sparse = dict(processed=True, dir='/media/donghee/kradar/rdr_sparse_data/rtnh_wider_1p_1',),
    rdr_polar_3d = dict(processed=True, dir='/media/donghee/kradar/rdr_polar_3d'),
    portion = ['1'], # ['7', '8'],
)

class GetDopplerPoints(object):
    def __init__(self, cfg=None):
        if cfg == None:
            cfg = EasyDict(dict_cfg)
            self.cfg=cfg
        else:
            self.cfg=cfg.DATASET

        self.label = self.cfg.label
        self.label_version = self.cfg.get('label_version', 'v2_0')
        
        self.item = self.cfg.item
        self.calib = self.cfg.calib
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.rdr_polar_3d = self.cfg.get('rdr_polar_3d', None)

        self.portion = self.cfg.get('portion', None)
        self.list_dict_item = self.load_dict_item(self.cfg.path_data)

        self.arr_range, self.arr_azimuth, self.arr_elevation, \
            self.arr_doppler = self.load_physical_values(is_with_doppler=True)

    def load_dict_item(self, path_data):
        list_seqs_w_header = []
        for path_header in path_data.list_dir_kradar:
            list_seqs = os.listdir(path_header)
            if self.portion is None:
                list_seqs_w_header.extend([(seq, path_header) for seq in list_seqs])
            else:
                for seq in list_seqs:
                    if seq in self.portion:
                        list_seqs_w_header.append((seq, path_header))
        list_seqs_w_header = sorted(list_seqs_w_header, key=lambda x: int(x[0]))

        list_dict_item = []
        for seq, path_header in list_seqs_w_header:
            list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
            for label in list_labels:
                path_label_v2_0 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_0', 'KRadar_refined_label_by_UWIPL', seq, label)
                dict_item = dict(
                    meta = dict(
                        header = path_header, seq = seq,
                        label_v2_0 = path_label_v2_0,
                    ),
                )
                dict_item = self.get_label(dict_item)
                list_dict_item.append(dict_item)
        return list_dict_item
    
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
            front = osp.join(header, seq, 'cam-front', f'cam-front_{camf}.png'),
            left = osp.join(header, seq, 'cam-left', f'cam-left_{camr}.png'),
            right = osp.join(header, seq, 'cam-right', f'cam-right_{camr}.png'),
            rear = osp.join(header, seq, 'cam-rear', f'cam-rear_{camr}.png'),
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
    
    def get_rdr_sparse(self, dict_item):
        dir_rdr_sparse = self.rdr_sparse.dir
        seq = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr']
        path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'sprdr_{rdr_idx}.npy')
        rdr_sparse = np.load(path_rdr_sparse)
        dict_item['rdr_sparse'] = rdr_sparse

        return dict_item

    def get_rdr_polar_3d(self, dict_item):
        cfg_rdr_polar_3d = self.rdr_polar_3d
        dict_meta = dict_item['meta']
        seq_name = dict_meta['seq']
        rdr_idx = dict_meta['idx']['rdr']
        path_polar_3d = osp.join(cfg_rdr_polar_3d.dir, seq_name, f'polar3d_{rdr_idx}.npy')
        cube_polar = np.load(path_polar_3d)
        
        dict_item['rdr_polar_3d'] = cube_polar
        
        return dict_item # 2, 256, 107, 37 (pw is normalized with 1e+13)

    def __len__(self):
        return len(self.list_dict_item)
    
    def __getitem__(self, idx):
        dict_item = self.list_dict_item[idx]
        dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
        dict_item = self.get_rdr_polar_3d(dict_item) if self.item['rdr_polar_3d'] else dict_item
        dict_item = self.get_rdr_sparse(dict_item) if self.item['rdr_sparse'] else dict_item
        dict_item = self.get_description(dict_item)
        
        return dict_item
    
    def save_doppler_points_for_rdr_sparse(self, dir_save='/media/donghee/HDD_3/rdr_sparse_doppler', vis=False):
        cfg_rdr_sparse = self.rdr_sparse
        rdr_sparse_name = (cfg_rdr_sparse.dir).split('/')[-1]
        dir_rdr_sparse_doppler = osp.join(dir_save, rdr_sparse_name)

        for seq_name in range(58):
            seq_folder = osp.join(dir_rdr_sparse_doppler, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)
        
        for idx_sample in tqdm(range(len(self))):
            dict_item = self.__getitem__(idx_sample)
            
            rdr_sparse = dict_item['rdr_sparse']
            rdr_polar_3d = dict_item['rdr_polar_3d']

            ### rdr_sparse to polar ###
            r = np.sqrt(np.sum(np.power(rdr_sparse[:,:3],2),axis=1))
            norm_one_xyz = rdr_sparse[:,:3]/(np.expand_dims(r,-1))
            el = -np.arcsin(norm_one_xyz[:,2]) # N,
            az = -np.arctan2(norm_one_xyz[:,1], norm_one_xyz[:,0]) # N,
            ### rdr_sparse to polar ###

            ### rdr polar to cartesian ###
            # r_ind, a_ind, e_ind = np.where(rdr_polar_3d[0,:,:,:]>np.quantile(rdr_polar_3d[0,:,:,:], 0.99))
            # rdr_points_from_3d = rdr_polar_3d[:,r_ind,a_ind,e_ind]
            # r = self.arr_range[r_ind]
            # az = -self.arr_azimuth[a_ind]
            # el = -self.arr_elevation[e_ind]

            # x = r * np.cos(el) * np.cos(az)
            # y = r * np.cos(el) * np.sin(az)
            # z = r * np.sin(el)
            # rdr_pc = np.stack((x,y,z), axis=1)
            ### rdr polar to cartesian ###

            ### rdr_sparse (polar) to index ###
            arr_r = self.arr_range
            arr_a = self.arr_azimuth
            arr_e = self.arr_elevation
            r_min = np.min(arr_r)
            r_bin = np.mean(arr_r[1:]-arr_r[:-1])
            a_min = np.min(arr_a)
            a_bin = np.mean(arr_a[1:]-arr_a[:-1])
            e_min = np.min(arr_e)
            e_bin = np.mean(arr_e[1:]-arr_e[:-1])

            r_ind = (np.clip(np.around((r-r_min)/r_bin), 0., 255.)).astype(np.int64)
            a_ind = (np.clip(np.around((az-a_min)/a_bin), 0., 106.)).astype(np.int64)
            e_ind = (np.clip(np.around((el-e_min)/e_bin), 0., 36.)).astype(np.int64)

            dop_val = rdr_polar_3d[1,r_ind,a_ind,e_ind]
            ### rdr_sparse (polar) to index ###

            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            rdr_idx = dict_meta['idx']['rdr']
            
            path_rdr_sparse_doppler = osp.join(dir_rdr_sparse_doppler, seq_name, f'rdr_sparse_doppler_{rdr_idx}.npy')
            rdr_sparse_doppler = np.concatenate((rdr_sparse, dop_val.reshape(-1,1)), axis=-1)
            np.save(path_rdr_sparse_doppler, rdr_sparse_doppler)

            for k in dict_item.keys():
                if k != 'meta':
                    dict_item[k] = None

            if vis:
                vis = o3d.visualization.Visualizer()
                vis.create_window()

                pc_lidar = dict_item['ldr64']
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
                vis.add_geometry(pcd)

                pcd_rdr = o3d.geometry.PointCloud()
                pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
                pcd_rdr.paint_uniform_color([0.,0.,0.])
                vis.add_geometry(pcd_rdr)

                label = dict_item['meta']['label']
                for obj in label:
                    cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                    consider, logit_idx, rgb, bgr = self.label[cls_name]
                    if consider:
                        self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
                vis.run()
                vis.destroy_window()

    def get_inside_points_indices(self, pc, list_bbox, is_vis=False):
        if is_vis:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
            vis.add_geometry(pcd)

            for obj in list_bbox:
                x, y, z, th, l, w, h = obj
                self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[1,0,0], radius=0.05)
            
            vis.run()
            vis.destroy_window()

        list_pc_indices = []
        for obj in list_bbox:
            x, y, z, th, l, w, h = obj
            rotation_matrix = np.array([
                [np.cos(-th), -np.sin(-th), 0],
                [np.sin(-th), np.cos(-th), 0],
                [0, 0, 1]
            ])

            rotated = np.dot(pc[:,:3]-np.array([[x,y,z]]), rotation_matrix.T)
            pc_ind = np.where(
                (rotated[:,0]>-l/2.) & (rotated[:,0]<l/2.) &
                (rotated[:,1]>-w/2.) & (rotated[:,1]<w/2.) &
                (rotated[:,2]>-h/2.) & (rotated[:,2]<h/2.))[0]
            list_pc_indices.append(pc_ind)
            
            if is_vis:
                vis = o3d.visualization.Visualizer()
                vis.create_window()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[pc_ind,:][:,:3])
                vis.add_geometry(pcd)

                vis.run()
                vis.destroy_window()
        
        return list_pc_indices

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
    ### Vis ###

if __name__ == '__main__':
    get_doppler_points = GetDopplerPoints()
    # get_doppler_points.save_doppler_points_for_rdr_sparse()
    
    dict_item = get_doppler_points[0]
    list_bbox = [vals for _, vals, _, _ in dict_item['meta']['label']]
    get_doppler_points.get_inside_points_indices(dict_item['ldr64'], list_bbox, is_vis=True)
    
    rpc = dict_item['rdr_sparse']
    list_points_ind = get_doppler_points.get_inside_points_indices(rpc, list_bbox)

    for ind_bbox in list_points_ind:
        rpc_cropped = rpc[ind_bbox, :]
        print(rpc_cropped.shape)
        print(np.mean(rpc_cropped, axis=0))
