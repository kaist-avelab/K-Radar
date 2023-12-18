'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import sys
import numpy as np
import open3d as o3d
import random
import torch
import nms
import pickle
import matplotlib.pyplot as plt

from scipy.io import loadmat # from matlab
from easydict import EasyDict
from tqdm import tqdm

# For executing the '__main__' directly
try:
    from utils.util_config import cfg_from_yaml_file
    from models.skeletons import build_skeleton
    from utils.Rotated_IoU.oriented_iou_loss import cal_iou
except:
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from utils.util_config import cfg_from_yaml_file
    from models.skeletons import build_skeleton
    from utils.Rotated_IoU.oriented_iou_loss import cal_iou

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

class RadarFilmDataProcessor(object):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = dict_cfg
        cfg = EasyDict(cfg)
        self.cfg = cfg

        list_dict_item = []
        rdr_dir = self.cfg.dir_data.rdr
        self.rdr_dir = rdr_dir
        self.ldr_dir = self.cfg.dir_data.ldr
        is_filter_for_valid_ldr = self.cfg.is_filter_for_valid_ldr
        include_seqs = self.cfg.include_seqs
        list_folders = sorted(os.listdir(rdr_dir), key=lambda x: int(x))
        for temp_seq in list_folders:
            if include_seqs is None:
                pass
            elif not (temp_seq in include_seqs):
                continue
            temp_rt_dir = osp.join(rdr_dir, temp_seq, 'radar_tensor')
            list_rt = sorted(os.listdir(temp_rt_dir))
            for rt in list_rt:
                path_rt = osp.join(rdr_dir, temp_seq, 'radar_tensor', rt)
                rdr_idx = path_rt.split(osp.sep)[-1].split('.')[0].split('_')[1]
                temp_dict = dict()
                temp_dict_idx = dict(rdr=rdr_idx)
                temp_dict['meta'] = dict(rt=path_rt, seq=temp_seq, idx=temp_dict_idx)
                self.get_proper_lpc_from_rdr_idx(temp_dict)
                if (temp_dict['meta']['lpc'] is None) and is_filter_for_valid_ldr:
                    continue # filter dict only if proper lpc exists
                list_dict_item.append(temp_dict)
        self.list_dict_item = list_dict_item
        self.arr_range, self.arr_azimuth, self.arr_elevation, \
            self.arr_doppler = self.load_physical_values(is_with_doppler=True)

    def __len__(self):
        return len(self.list_dict_item)
    
    def __getitem__(self, idx):
        dict_item = self.list_dict_item[idx]
        return dict_item
    
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
    
    def get_tesseract(self, dict_item):
        arr_tesseract = loadmat(dict_item['meta']['rt'])['arrDREA']
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
    
    def get_rdr_points_from_npy(self, dict_item, key='1p'):
        seq_name = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr']
        
        path_rdr = osp.join(self.rdr_dir, seq_name, f'rpc_{key}', f'rpc_{rdr_idx}.npy')
        dict_item['rdr_pc'] = np.load(path_rdr)

        return dict_item

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

    def get_proper_lpc_from_rdr_idx(self, dict_item):
        rdr_idx = int(dict_item['meta']['idx']['rdr'])
        seq_name = dict_item['meta']['seq']
        diff, rate = self.cfg.time_calib_info[seq_name]
        ldr_idx = int(diff + rdr_idx*rate)
        if ldr_idx < 0:
            dict_item['meta']['idx']['ldr'] = None
            dict_item['meta']['lpc'] = None
        else:
            ldr_idx = (f'{ldr_idx}').zfill(5)
            path_lpc = osp.join(self.ldr_dir, seq_name, 'os2-128', f'os2-128_{ldr_idx}.pcd')
            if osp.exists(path_lpc):
                dict_item['meta']['idx']['ldr'] = ldr_idx
                dict_item['meta']['lpc'] = path_lpc
            else:
                dict_item['meta']['idx']['ldr'] = None
                dict_item['meta']['lpc'] = None
        return dict_item
    
    def gen_list_of_rpc_from_tesseract(self, list_rate=[0.01, 0.001], list_key=['1p', '01p']):
        for idx_dict, dict_item in enumerate(tqdm(self.list_dict_item)):
            if idx_dict < 920:
                continue
            dict_item = self.get_cube_polar(dict_item)
            cube_pw = dict_item['cube_pw_polar'] # Normalized with 1e+13
            cube_dop = dict_item['cube_dop_cartesian'] # Expectation w/ pw dist

            for idx_rate, rate in enumerate(list_rate):
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

                rpc = np.stack((x,y,z,pw,dop), axis=1)

                folder_name = f'rpc_{list_key[idx_rate]}'
                rdr_idx = dict_item['meta']['idx']['rdr']
                file_name = f'rpc_{rdr_idx}.npy'

                dir_rpc = osp.join(self.rdr_dir, dict_item['meta']['seq'], folder_name)
                os.makedirs(dir_rpc, exist_ok=True)
                
                path_rdr = osp.join(dir_rpc, file_name)
                np.save(path_rdr, rpc)
    
    def gen_inferred_results_from_ldr(self, model_loader):
        for dict_item in tqdm(self.list_dict_item):
            pc_lidar = get_ldr64(dict_item['meta']['lpc'], is_calib=True, calib_vals=rdr_film_processor.cfg.spatial_calib_info)
            pred_boxes, pred_scores, pred_labels = model_loader.infer_ldr_model(pc_lidar)
            seq_name = dict_item['meta']['seq']
            dir_inf = osp.join(self.ldr_dir, seq_name, 'inf_ldr')
            rdr_idx = dict_item['meta']['idx']['rdr']
            os.makedirs(dir_inf, exist_ok=True)
            path_txt = osp.join(dir_inf, f'{rdr_idx}.txt')
            
            f = open(path_txt, 'w')
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x, y, z, l, w, h, th = box
                f.write(f'{x:.6f},{y:.6f},{z:.6f},{l:.6f},{w:.6f},{h:.6f},{th:.6f},{score:.6f},{label:.6f}\n')
            f.close()

    def gen_label_for_single_object(self, first_minimum_range=10., minimum_iou=0.01):
        total_dict = dict()
        def save_list_targ_obj_for_frames(list_targ_obj, dict_item, total_dict):
            key_name = dict_item['meta']['seq']
            total_dict[key_name] = list_targ_obj

        list_checking_duplication = []
        is_find_targ_object = False
        targ_obj = None
        list_targ_obj_for_frames = []
        for idx_dict, dict_item in enumerate(tqdm(self.list_dict_item)):
            seq_name = dict_item['meta']['seq']
            if seq_name not in list_checking_duplication: # new sequence
                list_checking_duplication.append(seq_name)
                is_find_targ_object = False
                targ_obj = None

                if idx_dict > 0:
                    save_list_targ_obj_for_frames(list_targ_obj_for_frames,\
                                                  self.list_dict_item[idx_dict-1], total_dict)
                list_targ_obj_for_frames = []

                # if seq_name != '1': # for debugging
                #     break
                
            if idx_dict == (len(self.list_dict_item)-1):
                save_list_targ_obj_for_frames(list_targ_obj_for_frames,\
                                              self.list_dict_item[idx_dict-1], total_dict)

            dir_inf = osp.join(self.ldr_dir, seq_name, 'inf_ldr')
            rdr_idx = dict_item['meta']['idx']['rdr']
            path_txt = osp.join(dir_inf, f'{rdr_idx}.txt')

            f = open(path_txt, 'r')
            lines = f.readlines()
            f.close()

            list_objs = []
            for line in lines:
                obj = [float(x) for x in line.split(',')] # x, y, z, l, w, h, th, score, label
                list_objs.append(obj)

            if is_find_targ_object:
                x, y, z, l, w, h, th, score, label = targ_obj
                targ_info = torch.tensor([x, y, l, w, th])
                targ_info = targ_info.reshape(1,1,5)
                cand_info = torch.tensor(list_objs).reshape(-1,9)
                cand_info = torch.concat((cand_info[:,0:2],cand_info[:,3:5],cand_info[:,6:7]), dim=1)
                len_obj, _ = cand_info.shape
                if len(cand_info) == 0:
                    list_targ_obj_for_frames.append(None)
                    continue
                targ_info = targ_info.repeat(1,len_obj,1).cuda()
                cand_info = cand_info.reshape(1,-1,5)
                cand_info = cand_info.cuda()
                # print(targ_info, cand_info)
                iou, _, _, _ = cal_iou(targ_info, cand_info)
                
                is_find_object = False
                for idx_obj in range(len(iou[0])):
                    if iou[0][idx_obj] > minimum_iou:
                        is_find_object = True
                        found_idx_obj = idx_obj
                if is_find_object:
                    targ_obj = list_objs[found_idx_obj]
                    list_targ_obj_for_frames.append(targ_obj)
                else:
                    list_targ_obj_for_frames.append(None)
            else:
                for obj_info in list_objs:
                    x, y, z, l, w, h, th, score, label = obj_info
                    if x < first_minimum_range:
                        targ_obj = obj_info
                        list_targ_obj_for_frames.append(targ_obj)
                        is_find_targ_object = True
                if targ_obj is None:
                    list_targ_obj_for_frames.append(None)
        with open('./tools/radar_film/targ_obj.pickle', 'wb') as f:
            pickle.dump(total_dict, f)
    
    def get_pw_distribution(self, rpc_type='1p'):
        def get_avg_vals_in_3d_bbox(pc, bbox, mag=1.0, is_vis=False):
            if is_vis: # b4 rotate & trans
                vis = o3d.visualization.Visualizer()
                vis.create_window()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
                vis.add_geometry(pcd)
                
                x, y, z, l, w, h, th, score, label = bbox
                self.draw_3d_box_in_cylinder(vis, (x,y,z), th, l, w, h)
                
                vis.run()
                vis.destroy_window()

            ### Translation -> Rotation ###
            n_pc, dim = pc.shape
            x, y, z, l, w, h, th, score, label = bbox
            pc_translated = pc[:,:3] - np.array([x,y,z]).reshape(1,3).repeat(n_pc,0)
            cth = np.cos(th)
            sth = np.sin(th)
            rot = np.array([[cth,sth,0.],[-sth,cth,0.],[0.,0.,1.]])
            pc_rotated = (np.dot(rot, pc_translated.T)).T
            new_pc = np.concatenate((pc_rotated, pc[:,3:]), axis=1)
            l = mag*l
            w = mag*l
            h = mag*l
            new_pc = new_pc[np.where(
                (new_pc[:, 0] > -l/2.) & (new_pc[:, 0] < l/2.) &
                (new_pc[:, 1] > -w/2.) & (new_pc[:, 1] < w/2.) &
                (new_pc[:, 2] > -h/2.) & (new_pc[:, 2] < h/2.))]
            ### Translation -> Rotation ###

            if is_vis: # after rotate & trans
                vis = o3d.visualization.Visualizer()
                vis.create_window()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(new_pc[:,:3])
                vis.add_geometry(pcd)
                
                vis.run()
                vis.destroy_window()

            return np.mean(new_pc, axis=0)

        with open('./tools/radar_film/targ_obj.pickle', 'rb') as f:
            dict_targ_obj = pickle.load(f)
        
        list_seqs = []
        for idx_dict, dict_item in enumerate(self.list_dict_item):
            list_seqs.append(int(dict_item['meta']['seq']))
        list_seqs = np.unique(np.array(list_seqs)).tolist()
        list_seqs = [f'{x}' for x in list_seqs]

        list_range_vals_per_seq = [[] for _ in range(len(list_seqs))]
        list_pw_vals_per_seq = [[] for _ in range(len(list_seqs))]

        list_check_duplication = []
        frame_idx = 0
        for idx_dict, dict_item in enumerate(tqdm(self.list_dict_item)):
            seq_name = dict_item['meta']['seq']
            if seq_name not in list_check_duplication:
                list_check_duplication.append(seq_name)
                frame_idx = 0

            idx_seq = list_seqs.index(seq_name)
            list_objs = dict_targ_obj[seq_name]
            obj = list_objs[frame_idx]

            frame_idx += 1

            if obj is None:
                continue
            else:
                dir_rpc = osp.join(self.rdr_dir, dict_item['meta']['seq'], f'rpc_{rpc_type}')
                rdr_idx = dict_item['meta']['idx']['rdr']
                path_rdr = osp.join(dir_rpc, f'rpc_{rdr_idx}.npy')
                rpc = np.load(path_rdr)
                # pc_lidar = get_ldr64(dict_item['meta']['lpc'], is_calib=True, calib_vals=rdr_film_processor.cfg.spatial_calib_info)
                avg_vals = get_avg_vals_in_3d_bbox(rpc, obj)
                _, _, _, pw_avg, dop_avg = avg_vals
                x_cen = obj[0]
                list_range_vals_per_seq[idx_seq].append(x_cen)
                list_pw_vals_per_seq[idx_seq].append(10*np.log10(pw_avg*1e+13))
                # list_pw_vals_per_seq[idx_seq].append(pw_avg)
        
        for idx_seq in range(len(list_range_vals_per_seq)):
            # print(list_range_vals_per_seq[idx_seq])
            # print(list_pw_vals_per_seq[idx_seq])
            plt.plot(list_range_vals_per_seq[idx_seq], list_pw_vals_per_seq[idx_seq], label=list_seqs[idx_seq])

        dict_save = dict()
        for idx_seq in range(len(list_range_vals_per_seq)):
            seq_name = list_seqs[idx_seq]
            dict_save[seq_name] = (list_range_vals_per_seq[idx_seq], list_pw_vals_per_seq[idx_seq])
        
        with open('./tools/radar_film/pw_dist.pickle', 'wb') as f:
            pickle.dump(dict_save, f)

        plt.legend()
        plt.show()

    def get_pw_distribution_from_pickle(self, smoothing_size=21):
        def low_pass_filter(values, window_size):
            if window_size <= 1:
                return values

            filtered_values = []
            half_window = window_size // 2

            for i in range(len(values)):
                start = max(i - half_window, 0)
                end = min(i + half_window + 1, len(values))
                filtered_values.append(sum(values[start:end]) / (end - start))

            return filtered_values

        with open('./tools/radar_film/pw_dist.pickle', 'rb') as f:
            dict_pw_dist = pickle.load(f)

        dict_rgb = dict()
        for seq_name in list(dict_pw_dist.keys()):
            dict_rgb[seq_name] = (random.random(), random.random(), random.random())

        dict_seq_info = self.cfg.seq_info
        list_filter_seqs = self.cfg.filter_seqs
        for seq_name in list(dict_pw_dist.keys()):
            if seq_name not in list_filter_seqs:
                continue

            seq_info = dict_seq_info[seq_name]
            list_range, list_power = dict_pw_dist[seq_name]
            tuple_rgb = dict_rgb[seq_name]
            
            # sort
            list_power_sorted = sorted(list_power, key=lambda x: list_range[list_power.index(x)])
            list_range_sorted = sorted(list_range)

            if list_range_sorted[-1] > 19:
                list_power_sorted = list_power_sorted[:-1]
                list_range_sorted = list_range_sorted[:-1]
            
            # smooth
            list_power_smooth = low_pass_filter(list_power_sorted, smoothing_size)
            plt.plot(list_range_sorted, list_power_smooth, label=seq_info, color=tuple_rgb)
            plt.scatter(list_range_sorted, list_power_sorted, color=tuple_rgb, alpha=0.5)
            plt.text(list_range_sorted[-1], list_power_smooth[-1], seq_info, color=tuple_rgb)

        plt.legend()
        plt.show()

class ModelLoader(object):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = dict_cfg
        cfg = EasyDict(cfg)
        self.cfg = cfg

        seed = self.cfg.models.get('seed', None)
        if seed is not None:
            self.set_random_seed(seed)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def set_ldr_model(self):
        cfg_ldr_model = self.cfg.models.get('ldr_model', None)
        path_cfg = cfg_ldr_model.cfg

        from utils.util_config import cfg
        cfg = cfg_from_yaml_file(path_cfg, cfg)
        model = build_skeleton(cfg).cuda()
        pt_dict_model = torch.load(cfg_ldr_model.pt)
        model.load_state_dict(pt_dict_model, strict=False)
        model.eval()
        self.ldr_model = model

    def infer_ldr_model(self, pc_lidar):
        x_min, y_min, z_min, x_max, y_max, z_max = self.cfg.models.roi
        pc_lidar_input = pc_lidar[np.where(
            (pc_lidar[:, 0] > x_min) & (pc_lidar[:, 0] < x_max) &
            (pc_lidar[:, 1] > y_min) & (pc_lidar[:, 1] < y_max) &
            (pc_lidar[:, 2] > z_min) & (pc_lidar[:, 2] < z_max))]
        dict_input = dict()
        list_pc_lidar = torch.from_numpy(pc_lidar_input).float() # .cuda()
        dict_input['ldr64'] = list_pc_lidar
        batch_indices = torch.full((len(pc_lidar_input),), 0) # .cuda()
        dict_input['batch_indices_ldr64'] = batch_indices
        dict_input['batch_size'] = 1
        dict_input['gt_boxes'] = torch.zeros((1,1,8))

        dict_output = self.ldr_model(dict_input)
        pred_dicts = dict_output['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        if self.cfg.models.ldr_model.is_nms_output:
            pred_boxes, pred_scores, pred_labels = self.get_nms_results(pred_boxes, pred_scores, pred_labels)
        
        return pred_boxes, pred_scores, pred_labels

    def get_nms_results(self, pred_boxes, pred_scores, pred_labels, nms_thr=0.1, is_cls_agnostic=False):
        if is_cls_agnostic:
            for idx_cls in np.unique(pred_labels):
                pass # TODO
        else:
            xy_xlyl_th = np.concatenate((pred_boxes[:,0:2], pred_boxes[:,3:5], pred_boxes[:,6:7]), axis=1)
            list_tuple_for_nms = [[(x, y), (xl, yl), th] for (x, y, xl, yl, th) in xy_xlyl_th.tolist()]
            ind = nms.rboxes(list_tuple_for_nms, pred_scores.tolist(), nms_threshold=nms_thr)
            return pred_boxes[ind], pred_scores[ind], pred_labels[ind]

dict_cfg = dict(
    dir_data = dict(
        ldr = '/media/donghee/HDD_0/Radar_Film/gen_files',
        rdr = '/media/donghee/BED085ECD085AB6B/RadarFilmData',
    ),
    include_seqs = None, # ['14', '16', '31', '33', '35', '37', '42', '46', '48', '49'], # None
    is_filter_for_valid_ldr = True,
    spatial_calib_info = [-2.54, 0.3, 0.7],
    time_calib_info = { # 'seq': [diff, rate]
        '1': [-2, 2], # LiDAR is 20Hz, Radar is approximately 10Hz
        '2': [-2, 2],
        '10': [-2, 2],
        '12': [-2, 2],
        '14': [-2, 2],
        '16': [-2, 2],
        '30': [-2, 2],
        '31': [-2, 2],
        '32': [-2, 2],
        '33': [-2, 2],
        '34': [-2, 2],
        '35': [-2, 2],
        '36': [-2, 2],
        '37': [-2, 2],
        '40': [-2, 2],
        '42': [-2, 2],
        '44': [-2, 2],
        '46': [-2, 2],
        '48': [-2, 2],
        '49': [-2, 2],
    },
    seq_info = {
        '1': 'baseline1',
        '2': 'baseline2',
        '10': '4mm-rect/0.1T/Open', # '4_off',
        '12': '4mm-rect/0.1T/3V-200mA', # '4_on_3V_200mA',
        '14': '8mm-rect/0.1T/Open', # '5_off',
        '16': '8mm-rect/0.1T/3V-30mA', #'5_on_3V_30mA',
        '30': '2mm-rect/0.3T/Open', # '9_off',
        '31': '2mm-rect/0.3T/Open', # '9_off',
        '32': '9_on_2V_460mA',
        '33': '2mm-rect/0.3T/2V-460mA', # '9_on_2V_460mA',
        '34': '4mm-rect/0.3T/Open', # '10_off',
        '35': '10_off',
        '36': '10_on_2V_217mA',
        '37': '4mm-rect/0.3T/2V-217mA', # '10_on_2V_217mA',
        '40': '0.5mm-rect/0.125T/Open', # '11-2_off',
        '42': '11-2_off',
        '44': '0.8mm-round/0.5T/Open 1', # '13_off',
        '46': '0.8mm-round/0.5T/Open 2', # '14_off',
        '48': 'Without film 1',
        '49': 'Without film 2',
    },
    filter_seqs = ['10', '12', '14', '16', '31', '33', '34', '37', '40', '44', '46', '48', '49'],
    models = dict(
        roi = [0.,-16.,-2.,72.,16.,7.6],
        seed = 1,
        ldr_model = dict(
            cfg = './configs/cfg_PVRCNNPP.yml',
            pt = './pretrained/PVRCNNPP_wide_29.pt',
            is_nms_output = True,
            ),
    )
)

if __name__ == '__main__':
    rdr_film_processor = RadarFilmDataProcessor()
    print(len(rdr_film_processor))
    # dict_item = rdr_film_processor[0]

    ### Step 1. generate rpc from tesseract ###
    # rdr_film_processor.gen_list_of_rpc_from_tesseract()
    ### Step 1. generate rpc from tesseract ###

    ### Step 2. matching rdr & ldr ###
    # # (1) Gen from tesseract
    # # dict_item = rdr_film_processor.get_tesseract(dict_item)
    # # dict_item = rdr_film_processor.get_cube_polar(dict_item)
    # # dict_item = rdr_film_processor.get_portional_rdr_points_from_tesseract(dict_item, rate=0.01)

    # # (2) Get from npy
    # dict_item = rdr_film_processor.get_rdr_points_from_npy(dict_item, key='01p')

    # pc_lidar = get_ldr64(dict_item['meta']['lpc'], is_calib=True, calib_vals=rdr_film_processor.cfg.spatial_calib_info)
    
    # ### Visualization & Overlap ###
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # pcd_ldr = o3d.geometry.PointCloud()
    # pcd_ldr.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
    # vis.add_geometry(pcd_ldr)
    
    # pcd_rdr = o3d.geometry.PointCloud()
    # pcd_rdr.points = o3d.utility.Vector3dVector(dict_item['rdr_pc'][:,:3])
    # pcd_rdr.paint_uniform_color([0.,0.,0.])
    # vis.add_geometry(pcd_rdr)
    
    # vis.run()
    # vis.destroy_window()
    # ### Visualization & Overlap ###
    ### Step 2. matching rdr & ldr ###

    ### Step 3. infer ###
    # model_loader = ModelLoader()
    # model_loader.set_ldr_model()
    # rdr_film_processor.gen_inferred_results_from_ldr(model_loader)
    ### Step 3. infer ###

    ### Step 4. post-processing ###
    # rdr_film_processor.gen_label_for_single_object()
    ### Step 4. post-processing ###

    ### Step 5. get pw distribution ###
    # rdr_film_processor.get_pw_distribution()
    ### Step 5. get pw distribution ###

    ### Step 6. get smooth pw dist from pickle ###
    rdr_film_processor.get_pw_distribution_from_pickle()
    ### Step 6. get smooth pw dist from pickle ###
