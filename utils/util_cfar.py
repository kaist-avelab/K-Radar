'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
* description: CFAR algorithms
'''

import numpy as np
from tqdm import tqdm
from scipy import ndimage
from tqdm import tqdm
import open3d as o3d
import cv2

__all__ = [
    'CFAR',
]

class CFAR:
    def __init__(self, type='pointcloud', cfg=None):
        '''
        * type in ['pointcloud', 'index', 'both']
        '''
        self.cfg = cfg

        ### Design parameters ###
        # self.LARGE_VALUE = 1e+15
        self.grid_size = self.cfg.DATASET.RDR_CUBE.GRID_SIZE # [m]
        self.n_half_guard_cell_zyx = self.cfg.DATASET.RDR_CUBE.CFAR_PARAMS.GUARD_CELL
        self.n_half_train_cell_zyx = self.cfg.DATASET.RDR_CUBE.CFAR_PARAMS.TRAIN_CELL

        self.guard_cell_range_zyx = ((2*(np.array(self.n_half_guard_cell_zyx))+1)*self.grid_size).tolist()
        self.boundary_cell_range_zyx = ((2*(np.array(self.n_half_train_cell_zyx))+2*(np.array(self.n_half_guard_cell_zyx))+1)*self.grid_size).tolist()
        
        self.fa_rate = self.cfg.DATASET.RDR_CUBE.CFAR_PARAMS.FA_RATE # for CA-CFAR
        self.thr_rate = self.cfg.DATASET.RDR_CUBE.CFAR_PARAMS.THR_RATE # for OS-CFAR
        ### Design parameters ###

        self.roi = self.cfg.DATASET.RDR_CUBE.ROI
        arr_z_cb, arr_y_cb, arr_x_cb = self.roi['z'], self.roi['y'], self.roi['x']
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

    def ca_cfar(self, cube, cube_doppler):
        invalid_idx = np.where(cube==-1.)
        grid_size = self.grid_size

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
        z_min, y_min, x_min = self.min_values
        z_ind, y_ind, x_ind = pc_idx
        if self.cfg.DATASET.RDR_CUBE.GENERATE_SPARSE_CUBE.IS_ADD_HALF_GRID_OFFSET:
            # center of voxel, 0 indexed
            if self.cfg.DATASET.RDR_CUBE.GENERATE_SPARSE_CUBE.TYPE_OFFSET == 'plus':
                z_pc_coord = np.expand_dims((z_min + z_ind * grid_size) + grid_size / 2, axis=1)
                y_pc_coord = np.expand_dims((y_min + y_ind * grid_size) + grid_size / 2, axis=1)
                x_pc_coord = np.expand_dims((x_min + x_ind * grid_size) + grid_size / 2, axis=1)
            elif self.cfg.DATASET.RDR_CUBE.GENERATE_SPARSE_CUBE.TYPE_OFFSET == 'minus':
                z_pc_coord = np.expand_dims((z_min + z_ind * grid_size) - grid_size / 2, axis=1)
                y_pc_coord = np.expand_dims((y_min + y_ind * grid_size) - grid_size / 2, axis=1)
                x_pc_coord = np.expand_dims((x_min + x_ind * grid_size) - grid_size / 2, axis=1)
            else:
                print('* Exception error (Dataset): check GENERATE_SPARSE_CUBE.TYPE_OFFSET')
        else:
            # print('* debug here')
            z_pc_coord = np.expand_dims(z_min + z_ind * grid_size, axis=1)
            y_pc_coord = np.expand_dims(y_min + y_ind * grid_size, axis=1)
            x_pc_coord = np.expand_dims(x_min + x_ind * grid_size, axis=1)
        
        correp_power = np.expand_dims(correp_power, axis=1)
        list_attr = [x_pc_coord, y_pc_coord, z_pc_coord, correp_power]

        if cube_doppler is not None:
            correp_doppler = np.expand_dims(cube_doppler[pc_idx], axis=1)
            list_attr.append(correp_doppler)

        total_values = np.concatenate(list_attr, axis=1)
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

        ### To point cloud (TBD: update here) ###
        min_z, min_y, min_x = self.min_values
        indices_z, indices_y, indices_x = pc_idx
        pc_z = min_z + indices_z*self.grid_size + self.grid_size/2.
        pc_y = min_y + indices_y*self.grid_size + self.grid_size/2.
        pc_x = min_x + indices_x*self.grid_size + self.grid_size/2.
        
        total_values = np.concatenate([pc_x.reshape(-1,1), pc_y.reshape(-1,1), pc_z.reshape(-1,1), correp_power.reshape(-1,1)], axis=1)
        # fliter the power is -1.
        total_values = np.array(list(filter(lambda x: x[3] != -1., total_values.tolist())))
        ### To point cloud (TBD: update here) ###

        if self.mode == 0:
            return total_values # X, Y, Z, PW
        elif self.mode == 1:
            return pc_idx
        elif self.mode == 2:
            return total_values, pc_idx
