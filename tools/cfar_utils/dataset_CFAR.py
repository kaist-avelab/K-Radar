import os
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

import os.path as osp
from glob import glob
from scipy.io import loadmat # from matlab
import pickle
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d

from utils.util_geometry import Object3D

class CFARDataset(Dataset):
    def __init__(self):
        ### Pre-defined ###
        LIST_DIR_SEQ = ['/media/donghee/HDD_1/KRadar_for_eval/KRadar/SeqCar',\
                        '/media/donghee/HDD_2/KRadar_for_eval/KRadar/SeqCar2',
                        '/media/donghee/HDD_2/KRadar_for_eval/KRadar/SeqCyclist',\
                        '/media/donghee/HDD_2/KRadar_for_eval/KRadar/SeqPedestrian',\
                        '/media/donghee/HDD_1/KRadar_for_eval/KRadar/SeqBus']
        self.CLASS_ID = {
            'Sedan': 1,
            'Bus or Truck': 2,
            'Motorcycle': -1,
            'Bicycle': 3,
            'Bicycle Group': -1,
            'Pedestrian': 4,
            'Pedestrian Group': -1,
            'Background': 0,
        }
        self.Z_OFFSET = 1.25 # Radar to Lidar [m]
        # ROI = {
        #     'z': [-2, 5.6],
        #     'y': [-6.4, 6.0],
        #     'x': [0, 71.6],
        # } # For IITP evaluation
        ROI = {
            'z': [-2, 5.6],
            'y': [-12.8, 12.4],
            'x': [0, 71.6],
        }
        self.GET_ITEM = {
            'cube': True,
            'pc': True,
            'img': True
        }
        ### Pre-defined ###

        ### Consider ROIs ###
        self.arr_z_cb = np.arange(-30, 30, 0.4)
        self.arr_y_cb = np.arange(-80, 80, 0.4)
        self.arr_x_cb = np.arange(0, 100, 0.4)
        self.list_roi_idx_cb = [0, len(self.arr_z_cb)-1, \
            0, len(self.arr_y_cb)-1, 0, len(self.arr_x_cb)-1]
        idx_attr = 0
        for k, v in ROI.items():
            if v is not None:
                min_max = np.array(v).tolist()
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}_cb'), min_max)
                setattr(self, f'arr_{k}_cb', arr_roi)
                self.list_roi_idx_cb[idx_attr*2] = idx_min
                self.list_roi_idx_cb[idx_attr*2+1] = idx_max

                v_new = [arr_roi[0], arr_roi[-1]]
                v_new = np.array(v_new)
                ROI[k] = v_new
            idx_attr += 1
        self.roi = [ROI['x'][0], ROI['y'][0], ROI['z'][0],\
                    ROI['x'][1], ROI['y'][1], ROI['z'][1]]
        print('* Processed ROI: ', ROI)
        print('* Processed Indices: ', self.list_roi_idx_cb)
        ### Consider ROIs ###

        ### Get dictionries ###
        list_dict_datum = []
        for dir_seq in LIST_DIR_SEQ:
            list_seq = os.listdir(dir_seq)
            
            for name_seq in list_seq:
                path_seq = osp.join(dir_seq, name_seq)
                list_labels = sorted(glob(osp.join(path_seq, 'info_label', '*.txt')))
                
                list_pc_files, list_pc_tstamps = self.get_timestamps(osp.join(path_seq, 'info_frames', 'timestamp_pc.txt'))
                list_img_files, list_img_tstamps = self.get_timestamps(osp.join(path_seq, 'info_frames', 'timestamp_img.txt'))

                for path_label in list_labels:
                    dict_datum = dict()

                    idx_cube, idx_pcd = path_label.split('/')[-1].split('.')[0].split('_')
                    path_cube = osp.join(path_seq, 'radar_zyx_cube', f'cube_{idx_cube}.mat')
                    path_calib = osp.join(path_seq, 'info_matching', f'{idx_cube}_{idx_pcd}.txt')
                    
                    pc_file_name = f'pc_{idx_pcd}.pcd'
                    idx_pc_tstamp = list_pc_files.index(pc_file_name)
                    tstamp = list_pc_tstamps[idx_pc_tstamp]
                    dict_datum['timestamp'] = tstamp
                    idx_img_tstamp = np.argmin(np.abs(np.array(list_img_tstamps)-tstamp))
                    img_file_name = list_img_files[idx_img_tstamp]

                    path_pcd = osp.join(path_seq, 'lidar_point_cloud', pc_file_name)
                    path_img = osp.join(path_seq, 'front_image', img_file_name)
                    meta = {
                        'path_label': path_label,
                        'path_pcd': path_pcd,
                        'path_img': path_img,
                        'path_cube': path_cube,
                        'path_calib': path_calib,
                    }
                    dict_datum['meta'] = meta
                    dict_datum['calib'] = self.get_calib(path_calib)
                    dict_datum['label'] = self.get_bboxes(path_label, dict_datum['calib'])

                    list_dict_datum.append(dict_datum)
        self.list_dict_datum = list_dict_datum
        ### Get dictionries ###

    def get_config_values(self):
        dataset_config = dict()
        dataset_config['roi'] = self.get_roi_arrays('zyx')
        return dataset_config

    def get_roi_arrays(self, order='zyx'):
        if order=='zyx':
            ret = self.arr_z_cb, self.arr_y_cb, self.arr_x_cb
        elif order=='xyz':
            ret = self.arr_x_cb, self.arr_y_cb, self.arr_z_cb

        return ret

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        
        return arr[idx_min:idx_max+1], idx_min, idx_max

    def get_timestamps(self, path_timestamp):
        f = open(path_timestamp, 'r')
        lines = f.readlines()
        list_file_names = list(map(lambda x: x.split(',')[1], lines))
        list_timestamps = list(map(lambda x: float(x.split(',')[2]), lines))
        f.close()
        return list_file_names, list_timestamps
    
    def get_calib(self, path_calib):
        f = open(path_calib, 'r')
        lines = f.readlines()
        dx, dy, dz = list(map(lambda x: float(x), lines[1].split(',')))
        dz = self.Z_OFFSET
        return [dx, dy, dz]
    
    def get_bboxes(self, path_label, calib_info):
        with open(path_label, 'r') as f:
            lines = f.readlines()
            f.close()
        line_objects = lines[1:]
        list_objects = []

        for line in line_objects:
            temp_tuple = self.get_tuple_object(line, calib_info)
            if temp_tuple is not None:
                list_objects.append(temp_tuple)

        return list_objects

    def get_tuple_object(self, line, calib_info):
        list_values = line.split(',')

        if list_values[0] != '*':
            return None

        offset = 0
        if(len(list_values)) == 11:
            offset = 1
        cls_name = list_values[2+offset][1:]

        idx_cls = self.CLASS_ID[cls_name]

        idx_obj = int(list_values[1+offset])
        x = float(list_values[3+offset])
        y = float(list_values[4+offset])
        z = float(list_values[5+offset])
        theta = float(list_values[6+offset])
        theta = theta*np.pi/180. # radian
        l = 2*float(list_values[7+offset])
        w = 2*float(list_values[8+offset])
        h = 2*float(list_values[9+offset])
        
        x = x+calib_info[0]
        y = y+calib_info[1]
        z = z+calib_info[2]

        x_min, y_min, z_min, x_max, y_max, z_max = self.roi
        if ((x > x_min) and (x < x_max) and \
            (y > y_min) and (y < y_max) and \
            (z > z_min) and (z < z_max)):
            return (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
        else:
            return None

    def __len__(self):
        return len(self.list_dict_datum)

    def __getitem__(self, idx):
        dict_datum = self.list_dict_datum[idx].copy()

        dict_datum['cube'] = self.get_cube(dict_datum) if self.GET_ITEM['cube'] else None
        dict_datum['pc'] = self.get_pc(dict_datum) if self.GET_ITEM['pc'] else None
        dict_datum['img'] = self.get_img(dict_datum) if self.GET_ITEM['img'] else None

        return dict_datum

    def get_cube(self, dict_datum):
        arr_cube = np.flip(loadmat(dict_datum['meta']['path_cube'])['arr_zyx'], axis=0) # z-axis is flipped
        idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
        arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        
        # do this in collate fn
        # arr_cube = np.maximum(arr_cube, 0.) # remove -1

        return arr_cube

    def get_pc(self, dict_datum):
        pc_lidar = []
        with open(dict_datum['meta']['path_pcd'], 'r') as f:
            lines = [line.rstrip('\n') for line in f][13:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 4)

        ### Filter out missing values ###
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi
        x_min = x_min + 0.01
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] > x_min)].reshape(-1, 4)
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] < x_max)].reshape(-1, 4)
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 1] > y_min)].reshape(-1, 4)
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 1] < y_max)].reshape(-1, 4)
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 2] > z_min)].reshape(-1, 4)
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 2] < z_max)].reshape(-1, 4)
        # ### Filter out missing values ###

        calib_info = dict_datum['calib']
        pc_lidar = np.array(list(map(lambda x: \
                    [x[0]+calib_info[0], x[1]+calib_info[1], x[2]+calib_info[2], x[3]],\
                    pc_lidar.tolist())))

        return pc_lidar

    def get_img(self, dict_datum):
        path_img = dict_datum['meta']['path_img']
        img = cv2.imread(path_img)

        return img

    def collate_fn(self, samples):
        dict_batches = dict()

        list_cube = []
        list_label = []
        list_meta = []
        for sample in samples:
            cube = np.maximum(sample['cube'], 0) # remove -1
            list_cube.append(np.expand_dims(cube, 0))
            list_label.append(sample['label'])
            list_meta.append(sample['meta'])

        total_cube = np.concatenate(list_cube, axis=0)
        dict_batches['cube'] = torch.from_numpy(total_cube).float().cuda()
        dict_batches['label'] = list_label
        dict_batches['meta'] = list_meta

        return dict_batches

### For Visualization ###
def show_radar_cube(dict_datum, roi_arrays, is_y_axis_inverted=True):
    ### Parameters ###
    magnifying = 4 # for resolution
    dict_bbox_colors = {# BGR
        'Sedan': [0,0,255],
        'Bus or Truck': [0,255,0],
        'Bicycle': [0,255,255],
        'Pedestrian': [0,128,128]
    }
    line_thickness = 2
    alpha = 0.5
    ### Parameters ###

    rdr_cube = dict_datum['cube']
    arr_z_cb, arr_y_cb, arr_x_cb = roi_arrays
    
    ### Normalization ###
    n_z, n_y, n_x = rdr_cube.shape
    n_z_none_m1 = n_z-(np.count_nonzero(rdr_cube==-1., axis=0))
    n_z_none_m1 = np.maximum(n_z_none_m1, 1)
    rdr_cube = np.maximum(rdr_cube, 0)
    rdr_cube_bev = np.sum(rdr_cube, axis=0)
    rdr_cube_bev = rdr_cube_bev/n_z_none_m1
    ### Normalization ###

    ### Plot with meshgrid ###
    rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for vis
    arr_0, arr_1 = np.meshgrid(arr_x_cb, arr_y_cb)
    height, width = n_y, n_x
    figsize = (1*magnifying, height/width*magnifying) \
        if height>=width else (width/height*magnifying, 1*magnifying)
    plt.figure(figsize=figsize)
    plt.clf()
    plt.pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig('./resources/imgs/img_cube_bev.png', pad_inches=0, dpi=300)
    plt.close()
    # plt.show()
    ### Plot with meshgrid ###
    
    img_jet = cv2.imread('./resources/imgs/img_cube_bev.png')
    img_jet = np.flip(img_jet, axis=0)
    img_jet = cv2.resize(img_jet, (n_x*magnifying, n_y*magnifying), interpolation=cv2.INTER_LINEAR)
    img_alp = img_jet.copy()

    ### Drawing bounding boxes ###
    # Magnifying arr y & x
    n_y_interp = magnifying*n_y
    n_x_interp = magnifying*n_x
    arr_y_interp = np.linspace(np.min(arr_y_cb), np.max(arr_y_cb), num=n_y_interp)
    arr_x_interp = np.linspace(np.min(arr_x_cb), np.max(arr_x_cb), num=n_x_interp)
    m_per_pix_y = np.mean(arr_y_interp[1:]-arr_y_interp[:-1])
    m_per_pix_x = np.mean(arr_x_interp[1:]-arr_x_interp[:-1])
    min_y, max_y = np.min(arr_y_interp), np.max(arr_y_interp)
    min_x, max_x = np.min(arr_x_interp), np.max(arr_x_interp)
    
    for bbox in dict_datum['label']:
        cls_name, _, [x, y, z, theta, xl, yl, zl], _ = bbox # rad
        color = dict_bbox_colors[cls_name]
        rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        apex = np.transpose(np.array([[xl/2.,yl/2.],[-xl/2.,yl/2.],[-xl/2.,-yl/2.],[xl/2.,-yl/2.],[0,0],[xl/2,0]]))
        translation = np.repeat(np.array([[x],[y]]),6,axis=1)
        pts = np.transpose(translation+np.matmul(rot,apex))
        
        # meter to pix
        pts[:,0] = (pts[:,0]-min_x)/m_per_pix_x
        pts[:,1] = (pts[:,1]-min_y)/m_per_pix_y
        pts = pts.astype(int)
        # meter to pix
        
        apex = pts[:4,:]
        cen_to_dir = pts[4:,:]

        ### Drawing bbox ###
        img_jet = cv2.line(img_jet,(apex[0,0],apex[0,1]),(apex[1,0],apex[1,1]),color,thickness=line_thickness)
        img_jet = cv2.line(img_jet,(apex[1,0],apex[1,1]),(apex[2,0],apex[2,1]),color,thickness=line_thickness)
        img_jet = cv2.line(img_jet,(apex[2,0],apex[2,1]),(apex[3,0],apex[3,1]),color,thickness=line_thickness)
        img_jet = cv2.line(img_jet,(apex[3,0],apex[3,1]),(apex[0,0],apex[0,1]),color,thickness=line_thickness)

        img_jet = cv2.arrowedLine(img_jet,(cen_to_dir[0,0],cen_to_dir[0,1]),\
            (cen_to_dir[1,0],cen_to_dir[1,1]),color,thickness=line_thickness,tipLength=0.3)
        ### Drawing bbox ###
    
    ### Alpha blending ###
    img_jet = cv2.addWeighted(img_jet, alpha, img_alp, (1-alpha), 0)
    if is_y_axis_inverted:
        img_jet = np.flip(img_jet, axis=0)
    ### Alpha blending ###

    cv2.imshow('Radar Tensor in BEV', img_jet)
    cv2.waitKey(0)

def show_lidar_point_cloud(dict_datum, roi_arrays):
    ### Parameters ###
    dict_bbox_colors = {# RGB
        'Sedan': [1,0,0],
        'Bus or Truck': [0,1,0],
        'Bicycle': [1,1,0],
        'Pedestrian': [0,0.5,0.5]
    }
    ### Parameters ###

    pc_lidar = dict_datum['pc']
    arr_z_cb, arr_y_cb, arr_x_cb = roi_arrays
    # ROI filtering
    pc_lidar = pc_lidar[
        np.where(
            (pc_lidar[:, 0] > arr_x_cb[0]) & (pc_lidar[:, 0] < arr_x_cb[-1]) &
            (pc_lidar[:, 1] > arr_y_cb[0]) & (pc_lidar[:, 1] < arr_y_cb[-1]) &
            (pc_lidar[:, 2] > arr_z_cb[0]) & (pc_lidar[:, 2] < arr_z_cb[-1])
        )
    ]

    bboxes_o3d = []
    colors_line_o3d = []
    for obj in dict_datum['label']:
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
        # try item()
        bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta))

        lines = [[0, 1], [2, 3], #[0, 3],[1, 2], # front
                [4, 5], [6, 7], #[5, 6],[4, 7],
                [0, 4], [1, 5], [2, 6], [3, 7],
                [0, 2], [1, 3], [4, 6], [5, 7]]
        colors_bbox = [dict_bbox_colors[cls_name] for _ in range(len(lines))]
        colors_line_o3d.append(colors_bbox)

    line_sets_bbox = []
    for idx_bbox, gt_obj in enumerate(bboxes_o3d):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors_line_o3d[idx_bbox])
        line_sets_bbox.append(line_set)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # Display the bounding boxes:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])

    o3d.visualization.draw_geometries([pcd] + line_sets_bbox)

def show_camera_img(dict_datum):
    cv2.imshow('img', dict_datum['img'])
    cv2.waitKey(0)

def show_camera_img(dict_datum):
    cv2.imshow('img', dict_datum['img'])
    cv2.waitKey(0)

def save_power_distribution_inside_object(dict_datum, roi_arrays, folder_name):
    rdr_cube = dict_datum['cube'] # invalid = -1.
    arr_z_cb, arr_y_cb, arr_x_cb = roi_arrays
    
    ### Extracting voxels in ROI ###
    n_z, n_y, n_x = rdr_cube.shape
    m_per_pix_z = np.mean(arr_z_cb[1:]-arr_z_cb[:-1])
    m_per_pix_y = np.mean(arr_y_cb[1:]-arr_y_cb[:-1])
    m_per_pix_x = np.mean(arr_x_cb[1:]-arr_x_cb[:-1])
    min_z, max_z = np.min(arr_z_cb), np.max(arr_z_cb)
    min_y, max_y = np.min(arr_y_cb), np.max(arr_y_cb)
    min_x, max_x = np.min(arr_x_cb), np.max(arr_x_cb)

    for bbox in dict_datum['label']:
        cls_name, _, [x, y, z, theta, xl, yl, zl], _ = bbox # rad

        # Processing for XY
        rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        apex = np.transpose(np.array([[xl/2.,yl/2.],[-xl/2.,yl/2.],[-xl/2.,-yl/2.],[xl/2.,-yl/2.]]))
        translation = np.repeat(np.array([[x],[y]]),4,axis=1)
        pts = np.transpose(translation+np.matmul(rot,apex))
        # Processing for XY

        # Processing for Z
        pts = np.repeat(pts,2,axis=0)
        pts_z = np.transpose(np.repeat(np.array([[z+zl/2., z-zl/2.]]), 4)).reshape(-1,1)
        pts = np.concatenate([pts, pts_z], axis=1)
        # Processing for Z

        # pts to pix
        pts[:,0] = (pts[:,0]-min_x)/m_per_pix_x
        pts[:,1] = (pts[:,1]-min_y)/m_per_pix_y
        pts[:,2] = (pts[:,2]-min_z)/m_per_pix_z
        pts = pts.astype(int)
        # pts to pix

        # min max idx XYZ
        min_indices = np.min(pts, axis=0)
        max_indices = np.max(pts, axis=0)
        # min max idx XYZ

        # Slicing & filter unvalid
        sliced = rdr_cube[min_indices[2]:max_indices[2], min_indices[1]:max_indices[1], min_indices[0]:max_indices[0]].copy()

        # Coordinates
        sliced_indices_z = np.arange(min_indices[2], max_indices[2])
        sliced_indices_y = np.arange(min_indices[1], max_indices[1])
        sliced_indices_x = np.arange(min_indices[0], max_indices[0])

        num_z = len(sliced_indices_z)
        num_y = len(sliced_indices_y)
        num_x = len(sliced_indices_x)
        sliced_indices_z = np.tile(sliced_indices_z.reshape(-1,1,1,1), reps=(1,num_y,num_x,1))
        sliced_indices_y = np.tile(sliced_indices_y.reshape(1,-1,1,1), reps=(num_z,1,num_x,1))
        sliced_indices_x = np.tile(sliced_indices_x.reshape(1,1,-1,1), reps=(num_z,num_y,1,1))

        # Checking validity
        # pos_embeddings_idx = np.concatenate([sliced_indices_z, sliced_indices_y, sliced_indices_x, np.expand_dims(sliced, axis=3)], axis=3) # Z Y X sliced
        # for idx_z in range(num_z):
        #     for idx_y in range(num_y):
        #         for idx_x in range(num_x):
        #             from_pos_emb = pos_embeddings_idx[idx_z,idx_y,idx_x,:]
        #             arr_a = np.array([min_indices[2]+idx_z, min_indices[1]+idx_y, min_indices[0]+idx_x, sliced[idx_z,idx_y,idx_x]])
        #             arr_b = np.array(from_pos_emb)
        #             arr_bool = arr_a == arr_b
        #             for check in arr_bool.tolist():
        #                 if check == False:
        #                     print('False happens')
        # print(sliced.shape)
        # print(sliced_indices_z.shape)
        # print(sliced_indices_y.shape)
        # print(sliced_indices_x.shape)

        # idx to meter
        sliced_z = (min_z + sliced_indices_z*m_per_pix_z) + m_per_pix_z/2.
        sliced_y = (min_y + sliced_indices_y*m_per_pix_y) + m_per_pix_y/2.
        sliced_x = (min_x + sliced_indices_x*m_per_pix_x) + m_per_pix_x/2.
        
        pos_embeddings = np.concatenate([sliced_z, sliced_y, sliced_x, np.expand_dims(sliced, axis=3)], axis=3) # Z Y X sliced

        sliced = np.squeeze(pos_embeddings.reshape(-1, 1, 1, 4)).tolist() # (ZYX, 4)
        sliced = list(filter(lambda x: x[3]!=-1., sliced))
        num_voxels = len(sliced)
        cen_range = np.round(np.sqrt(x**2+y**2+z**2), decimals=6)
        # Slicing & filter unvalid

        sliced_array = np.array(sliced)

        sliced = sliced_array[:,3].tolist()
        pos = sliced_array[:,0:3].tolist()

        # save data
        temp_dict = dict()
        temp_dict['cls'] = cls_name
        temp_dict['cen_range'] = cen_range
        temp_dict['num_voxels'] = num_voxels
        temp_dict['valid_voxels'] = sliced
        temp_dict['pos'] = pos

        path_label = dict_datum['meta']['path_label'].split('/')
        seq_name = path_label[-3]
        file_name = (path_label[-1].split('.'))[0]
        save_path = f'./{folder_name}/{seq_name}-{file_name}-{cls_name}-{cen_range}.pickle'
        with open(save_path, 'wb') as f:
            pickle.dump(temp_dict, f)
        # save data

def show_tsne(care_obj = ['Sedan'], ):
    pass

def show_power_distribution_object(list_indices, list_color, folder_name, num_grid=200, is_with_rdr_norm=True):
    path_dir = f'./{folder_name}'
    list_files = sorted(os.listdir(path_dir))
    list_path = list(map(lambda x: os.path.join(path_dir, x), list_files))
    list_range = list(map(lambda x: float((x.split('-')[-1]).split('.')[0]), list_files))
    print(f'* ranges of center: {list_range}')

    list_vis_hist = []
    for idx in list_indices:
        with open(list_path[idx], 'rb') as f:
            dict_datum = pickle.load(f)
        cls_name = dict_datum['cls']
        cen_range = dict_datum['cen_range']
        num_voxles = dict_datum['num_voxels']
        pos = dict_datum['pos']
        list_vis_hist.append([dict_datum['valid_voxels'], f'{cls_name} at {cen_range} [m], # of bins = {num_voxles}', pos])
    
    fig = plt.figure(figsize=(8,5))
    fig.set_facecolor('white')
    for idx_temp, list_temp in enumerate(list_vis_hist):
        if is_with_rdr_norm:
            ### Radar Normalization ###
            pw = np.array(list_temp[0])
            label = list_temp[1]
            pos = list_temp[2]
            each_range = np.sqrt(np.sum(np.square(np.array(pos)),axis=1))
            
            ### distance ###
            # plt.hist(each_range.tolist(), num_grid, density=False,\
            #     color=list_color[idx_temp], alpha=0.75, edgecolor='k', label=list_temp[1])
            ### distance ###

            pw = np.multiply(pw, each_range**(2.2))

            pw_plot = (pw/(2e+16)).tolist()
            plt.hist(pw_plot, num_grid, density=True, #range=(0, 1.0),\
                color=list_color[idx_temp], alpha=0.5, edgecolor='k', label=label)
            ### Radar Normalization ###

        else: # Heuristic norm
            plt.hist((np.array(list_temp[0])/(1e+13)).tolist(), num_grid, density=False,\
                color=list_color[idx_temp], alpha=0.75, edgecolor='k', label=list_temp[1])

    plt.xlabel('power / (1e+13)')
    plt.ylabel('# of bins')
    plt.legend()
    plt.title('Histogram of power values')

    plt.show()
    plt.close()
### For Visualization ###

### For Checking Class Distribution ###
def checking_class_distribution(dataset, cls_names = ['Sedan', 'Bus or Truck', 'Bicycle', 'Pedestrian']):
    total_data_number = len(dataset)
    
    dict_cls = dict()
    for cls_name in cls_names:
        temp_dict = dict()
        temp_dict['num'] = 0
        temp_dict['value_sum'] = np.zeros(7) # x, y, z, theta, xl, yl, zl
        dict_cls[cls_name] = temp_dict

    for i in tqdm(range(total_data_number)):
        temp_datum = dataset[i]

        objects = temp_datum['label']

        for obj in objects:
            cls_name, _, list_values, _ = obj
            dict_cls[cls_name]['num'] += 1
            dict_cls[cls_name]['value_sum'] += np.array(list_values)

    for k, v in dict_cls.items():
        print('='*30)
        print(f'* For class {k}')
        list_mean_vals = v['value_sum']/v['num']
        x, y, z, theta, xl, yl, zl = list_mean_vals
        print(f'* x={x}, y={y}, z={z}, theta={theta}, xl={xl}, yl={yl}, zl={zl}')
        print('='*30)
### For Checking Class Distribution ###

if __name__ == '__main__':
    iitp_dataset = CFARDataset()
    print('* Total length of data: ', len(iitp_dataset))

    ### Visualization ###
    # for idx in tqdm(np.arange(50,500)):
    #     dict_datum = iitp_dataset[idx]
    #     print(dict_datum['label'])
    #     # show_radar_cube(dict_datum, iitp_dataset.get_roi_arrays('zyx'))
    #     # show_lidar_point_cloud(dict_datum, iitp_dataset.get_roi_arrays('zyx'))
    #     # show_camera_img(dict_datum)
    ### Visualization ###

    ### Generation ###
    # selected_idx = [60, 100, 250]
    for idx in tqdm(np.arange(0,1930)):
        # Skip indices
        # if idx in selected_idx:
        #     pass
        # else:
        #     continue

        dict_datum = iitp_dataset[idx]
        print(dict_datum['label'])
        save_power_distribution_inside_object(dict_datum,\
            iitp_dataset.get_roi_arrays('zyx'), folder_name='dict_power_dist_4')
    ### Generation ###

    ### For Showing Power Distribution ###
    # show_power_distribution_object(list_indices=[0, 2],\
    #     list_color=['g', 'r'], folder_name='dict_power_dist_2', is_with_rdr_norm=True)
    ### For Showing Power Distribution ###

    ### For Checking Class Distribution ###
    # checking_class_distribution(iitp_dataset)
    ### For Checking Class Distribution ###
    