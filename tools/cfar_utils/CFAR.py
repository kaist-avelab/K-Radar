import os
import numpy as np
from dataset_CFAR import CFARDataset
from tqdm import tqdm
from scipy import ndimage
from tqdm import tqdm
import open3d as o3d
import cv2

from utils.util_geometry import Object3D

class CFAR:
    def __init__(self, roi, type='pointcloud'):
        '''
        * type in ['pointcloud', 'index', 'both']
        '''

        ### Design parameters ###
        # self.LARGE_VALUE = 1e+15
        self.grid_size = 0.4 # [m]
        self.n_half_guard_cell_zyx = [1, 2, 4]
        self.n_half_train_cell_zyx = [4, 8, 16]

        self.guard_cell_range_zyx = ((2*(np.array(self.n_half_guard_cell_zyx))+1)*self.grid_size).tolist()
        self.boundary_cell_range_zyx = ((2*(np.array(self.n_half_train_cell_zyx))+2*(np.array(self.n_half_guard_cell_zyx))+1)*self.grid_size).tolist()
        
        self.fa_rate = 0.05 # [m] for CA-CFAR
        self.thr_rate = 0.02 # for OS-CFAR
        ### Design parameters ###

        self.roi = roi
        arr_z_cb, arr_y_cb, arr_x_cb = self.roi
        self.min_values = [np.min(arr_z_cb), np.min(arr_y_cb), np.min(arr_x_cb)]

        ### Return mode ###
        if type == 'pointcloud':
            self.mode = 0
        elif type == 'index':
            self.mode = 1
        elif type == 'both':
            self.mode = 2
        ### Return mode ###
        
    def __str__(self):
        desc = f'* Considering total {self.boundary_cell_range_zyx}[m] (Z, Y, X) for tensor whose grid size {self.grid_size}[m].\n'
        desc += f'* Guard cell (Z, Y, X) = {self.guard_cell_range_zyx}[m]'
        return desc
    
    def fixed_points(self, cube, top_percent=0.1):
        cube_fix = cube.copy()
        pc_idx = np.where(cube_fix > np.quantile(cube_fix, 1-top_percent))
        correp_power = cube[pc_idx] # Unnormalized

        ### To point cloud ###
        min_z, min_y, min_x = self.min_values
        indices_z, indices_y, indices_x = pc_idx
        pc_z = min_z + indices_z*self.grid_size + self.grid_size/2.
        pc_y = min_y + indices_y*self.grid_size + self.grid_size/2.
        pc_x = min_x + indices_x*self.grid_size + self.grid_size/2.
        
        total_values = np.concatenate([pc_x.reshape(-1,1), pc_y.reshape(-1,1), pc_z.reshape(-1,1), correp_power.reshape(-1,1)], axis=1)
        # fliter the power is -1.
        total_values = np.array(list(filter(lambda x: x[3] != -1., total_values.tolist())))
        ### To point cloud ###

        if self.mode == 0:
            return total_values # X, Y, Z, PW
        elif self.mode == 1:
            return pc_idx
        elif self.mode == 2:
            return total_values, pc_idx

    def ca_cfar(self, cube):
        invalid_idx = np.where(cube==-1.)

        # normalize cube & make power level realy high or (set as representive value: mean) to invalid idx to make threshold high to edge part
        cube_norm = cube.copy()
        cube_norm[invalid_idx] = 0
        cube_norm = (cube_norm)/1e+13
        # cube_norm[invalid_idx] = self.LARGE_VALUE
        cube_norm[invalid_idx] = np.mean(cube_norm)
        
        # generating 3D mask
        nh_g_z, nh_g_y, nh_g_x = self.n_half_guard_cell_zyx
        nh_t_z, nh_t_y, nh_t_x = self.n_half_train_cell_zyx 
        mask_size = (2*(nh_g_z+nh_t_z)+1, 2*(nh_g_y+nh_t_y)+1, 2*(nh_g_x+nh_t_x)+1) # 1 for own
        mask = np.ones(mask_size)
        mask[nh_t_z:nh_t_z+2*nh_g_z+1, nh_t_y:nh_t_y+2*nh_g_y+1, nh_t_x:nh_t_x+2*nh_g_x+1] = 0
        num_total_train_cells = np.count_nonzero(mask)
        mask = mask/num_total_train_cells

        alpha = num_total_train_cells * (self.fa_rate**(-1/num_total_train_cells)-1)
        
        conv_out = ndimage.convolve(cube_norm, mask, mode='mirror')
        conv_out = alpha * conv_out

        out = np.greater(cube_norm, conv_out)
        pc_idx = np.where(out==True)
        correp_power = cube[pc_idx] # Unnormalized

        ### To point cloud ###
        min_z, min_y, min_x = self.min_values
        indices_z, indices_y, indices_x = pc_idx
        pc_z = min_z + indices_z*self.grid_size + self.grid_size/2.
        pc_y = min_y + indices_y*self.grid_size + self.grid_size/2.
        pc_x = min_x + indices_x*self.grid_size + self.grid_size/2.
        
        total_values = np.concatenate([pc_x.reshape(-1,1), pc_y.reshape(-1,1), pc_z.reshape(-1,1), correp_power.reshape(-1,1)], axis=1)
        # fliter the power is -1.
        total_values = np.array(list(filter(lambda x: x[3] != -1., total_values.tolist())))
        ### To point cloud ###

        if self.mode == 0:
            return total_values # X, Y, Z, PW
        elif self.mode == 1:
            return pc_idx
        elif self.mode == 2:
            return total_values, pc_idx
    
    def os_cfar(self, cube):
        invalid_idx = np.where(cube==-1.)

        # normalize cube & make power level realy high or (set as representive value: mean) to invalid idx to make threshold high to edge part
        cube_norm = cube.copy()
        cube_norm[invalid_idx] = 0
        cube_norm = (cube_norm)/1e+13
        # cube_norm[invalid_idx] = self.LARGE_VALUE
        cube_norm[invalid_idx] = np.mean(cube_norm)

        # generating 3D mask
        nh_g_z, nh_g_y, nh_g_x = self.n_half_guard_cell_zyx
        nh_t_z, nh_t_y, nh_t_x = self.n_half_train_cell_zyx

        margin_z = nh_g_z+nh_t_z
        margin_y = nh_g_y+nh_t_y
        margin_x = nh_g_x+nh_t_x

        out = np.zeros_like(cube)

        n_z, n_y, n_x = out.shape

        for idx_z in range(margin_z,n_z-margin_z):
            for idx_y in range(margin_y,n_y-margin_y):
                for idx_x in range(margin_x,n_x-margin_x):
                    mask = cube_norm[idx_z-margin_z:idx_z+margin_z+1,idx_y-margin_y:idx_y+margin_y+1,idx_x-margin_x:idx_x+margin_x+1].copy()
                    mask[nh_t_z:nh_t_z+2*nh_g_z+1, nh_t_y:nh_t_y+2*nh_g_y+1, nh_t_x:nh_t_x+2*nh_g_x+1] = -1
                    arr = mask[np.where(mask!=-1.)]
                    thr = np.quantile(arr, 1-self.thr_rate)
                    out[idx_z,idx_y,idx_x] = 1 if cube_norm[idx_z,idx_y,idx_x] > thr else 0

        pc_idx = np.where(out==1)
        correp_power = cube[pc_idx] # Unnormalized

        ### To point cloud ###
        min_z, min_y, min_x = self.min_values
        indices_z, indices_y, indices_x = pc_idx
        pc_z = min_z + indices_z*self.grid_size + self.grid_size/2.
        pc_y = min_y + indices_y*self.grid_size + self.grid_size/2.
        pc_x = min_x + indices_x*self.grid_size + self.grid_size/2.
        
        total_values = np.concatenate([pc_x.reshape(-1,1), pc_y.reshape(-1,1), pc_z.reshape(-1,1), correp_power.reshape(-1,1)], axis=1)
        # fliter the power is -1.
        total_values = np.array(list(filter(lambda x: x[3] != -1., total_values.tolist())))
        ### To point cloud ###

        if self.mode == 0:
            return total_values # X, Y, Z, PW
        elif self.mode == 1:
            return pc_idx
        elif self.mode == 2:
            return total_values, pc_idx

def show_pointcloud_for_lidar_and_radar(lpc, rpc, show='both', label=None):
    '''
    * lpc np.array
    * rpc np.array
    * show in ['lidar', 'radar', 'both']
    '''
    print('* label: ', label)
    print('* # of lpc: ', len(lpc))
    print('* # of rpc: ', len(rpc))
    ### Parameters ###
    dict_bbox_colors = {# RGB
        'Sedan': [1,0,0],
        'Bus or Truck': [0,1,0],
        'Bicycle': [1,1,0],
        'Pedestrian': [0,0.5,0.5]
    }
    color_rpc = [0, 0, 0]
    ### Parameters ###

    pcd_lpc = o3d.geometry.PointCloud()
    pcd_lpc.points = o3d.utility.Vector3dVector(lpc[:,:3])
    pcd_rpc = o3d.geometry.PointCloud()
    pcd_rpc.points = o3d.utility.Vector3dVector(rpc[:,:3])
    color = np.repeat(np.array(color_rpc).reshape(1,3), repeats=len(rpc), axis=0)
    pcd_rpc.colors = o3d.utility.Vector3dVector(color)
    list_vis = [pcd_lpc, pcd_rpc]

    if not (label is None):
        bboxes_o3d = []
        colors_line_o3d = []
        for obj in label:
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
        
        list_vis += line_sets_bbox

    o3d.visualization.draw_geometries(list_vis)



if __name__ == '__main__':
    iitp_dataset = CFARDataset()
    print('* Total length of data: ', len(iitp_dataset))

    cfar = CFAR(iitp_dataset.get_roi_arrays('zyx'), type='pointcloud')
    # print(cfar)
    
    for idx in tqdm(np.arange(1500,1900)):
        dict_datum = iitp_dataset[idx]
        
        cube = dict_datum['cube']
        # rpc = cfar.ca_cfar(cube)
        rpc = cfar.os_cfar(cube)
        # rpc = cfar.fixed_points(cube)
        
        label = dict_datum['label']
        print(dict_datum['img'].shape)
        cv2.imshow('image', dict_datum['img'])
        cv2.waitKey(0)

        show_pointcloud_for_lidar_and_radar(dict_datum['pc'], rpc, label=label)

        # exit()