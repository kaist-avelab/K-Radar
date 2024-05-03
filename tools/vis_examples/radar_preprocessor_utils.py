'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import numpy as np
import cv2
import open3d as o3d

__all__ = [
    'get_xy_from_ra_color',
    'PointCloudPcd',
    'RenderObject3D',
    'ObjectRenderer',
]

def get_xy_from_ra_color(ra_in, arr_range_in, arr_azimuth_in, roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_in_deg=False):
    '''
    * args:
    *   roi_x = [min_x, bin_x, max_x] [m]
    *   roi_y = [min_y, bin_y, max_y] [m]
    '''
    
    if len(ra_in.shape) == 2:
        num_range, num_azimuth = ra_in.shape
    elif len(ra_in.shape) == 3:
        num_range, num_azimuth, _ = ra_in.shape

    assert (num_range == len(arr_range_in) and num_azimuth == len(arr_azimuth_in))
    
    ra = ra_in.copy()
    arr_range = arr_range_in.copy()
    arr_azimuth = arr_azimuth_in.copy()

    if is_in_deg:
        arr_azimuth = arr_azimuth*np.pi/180.
    
    min_x, bin_x, max_x = roi_x
    min_y, bin_y, max_y = roi_y

    arr_x = np.linspace(min_x, max_x-bin_x, int((max_x-min_x)/bin_x))+bin_x/2.
    arr_y = np.linspace(min_y, max_y-bin_y, int((max_y-min_y)/bin_y))+bin_x/2.

    max_r = np.max(arr_range)
    min_r = np.min(arr_range)

    max_azi = np.max(arr_azimuth)
    min_azi = np.min(arr_azimuth)

    num_y = len(arr_y)
    num_x = len(arr_x)
    
    arr_yx = np.zeros((num_y, num_x, 3), dtype=np.uint8)

    # Inverse warping
    for idx_y, y in enumerate(arr_y):
        for idx_x, x in enumerate(arr_x):
            # bilinear interpolation

            r = np.sqrt(x**2 + y**2)
            azi = np.arctan2(-y,x) # for real physical azimuth
            
            if (r < min_r) or (r > max_r) or (azi < min_azi) or (azi > max_azi):
                continue
            
            try:
                idx_r_0, idx_r_1 = find_nearest_two(r, arr_range)
                idx_a_0, idx_a_1 = find_nearest_two(azi, arr_azimuth)
            except:
                continue

            if (idx_r_0 == -1) or (idx_r_1 == -1) or (idx_a_0 == -1) or (idx_a_1 == -1):
                continue
            
            ra_00 = ra[idx_r_0,idx_a_0,:]
            ra_01 = ra[idx_r_0,idx_a_1,:]
            ra_10 = ra[idx_r_1,idx_a_0,:]
            ra_11 = ra[idx_r_1,idx_a_1,:]

            val = (ra_00*(arr_range[idx_r_1]-r)*(arr_azimuth[idx_a_1]-azi)\
                    +ra_01*(arr_range[idx_r_1]-r)*(azi-arr_azimuth[idx_a_0])\
                    +ra_10*(r-arr_range[idx_r_0])*(arr_azimuth[idx_a_1]-azi)\
                    +ra_11*(r-arr_range[idx_r_0])*(azi-arr_azimuth[idx_a_0]))\
                    /((arr_range[idx_r_1]-arr_range[idx_r_0])*(arr_azimuth[idx_a_1]-arr_azimuth[idx_a_0]))

            arr_yx[idx_y, idx_x] = val

    return arr_yx, arr_y, arr_x

def find_nearest_two(value, arr):
    '''
    * args
    *   value: float, value in arr
    *   arr: np.array
    * return
    *   idx0, idx1 if is_in_arr else -1
    '''
    arr_temp = arr - value
    arr_idx = np.argmin(np.abs(arr_temp))
    
    try:
        if arr_temp[arr_idx] < 0: # min is left
            if arr_temp[arr_idx+1] < 0:
                return -1, -1
            return arr_idx, arr_idx+1
        elif arr_temp[arr_idx] >= 0:
            if arr_temp[arr_idx-1] >= 0:
                return -1, -1
            return arr_idx-1, arr_idx
    except:
        return -1, -1
    
class PointCloudPcd():
    def __init__(self, path_pcd:str, len_header:int=11, ego_offset:float=1e-3)->object:
        f = open(path_pcd, 'r')
        lines = f.readlines()
        f.close()
        self.path_pcd = path_pcd

        list_header = lines[:len_header]
        list_values = lines[len_header:]
        list_values = list(map(lambda x: x.split(' '), list_values))
        values = np.array(list_values, dtype=np.float32)
        values = values[ # delete (0,0)
            np.where(
                (values[:,0]<-ego_offset) | (values[:,0]>ego_offset) |  # x
                (values[:,1]<-ego_offset) | (values[:,1]>ego_offset)    # y
            )]
        self.values = values
        self.list_attr = (list_header[2].rstrip('\n')).split(' ')[1:]
        self.is_calibrated = False
        self.is_roi_filtered = False

    def __repr__(self)->str:
        str_repr = f'total {len(self.values)}x{len(self.list_attr)} points, fields = {self.list_attr}'
        if self.is_calibrated:
            str_repr += ', calibrated'
        if self.is_roi_filtered:
            str_repr += ', roi filtered'
        return str_repr
    
    @property
    def points(self): # x, y, z
        return self.values[:,:3]
    
    @property
    def points_w_attr(self):
        return self.values

    def _get_o3d_pcd(self)->o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd

    def _get_bev_pcd(self, dict_render)->np.array:
        x_min, x_bin, x_max = dict_render['ROI_X']
        y_min, y_bin, y_max = dict_render['ROI_Y']

        hue_type = dict_render['HUE']
        val_type = dict_render['VAL']

        pts_w_attr = (self.points_w_attr.copy()).tolist()
        pts_w_attr = np.array(sorted(pts_w_attr,key=lambda x: x[2])) # sort via z

        arr_x = np.linspace(x_min, x_max-x_bin, num=int((x_max-x_min)/x_bin)) + x_bin/2.
        arr_y = np.linspace(y_min, y_max-y_bin, num=int((y_max-y_min)/y_bin)) + y_bin/2.
        
        xy_mesh_grid_hsv = np.full((len(arr_x), len(arr_y), 3), 0, dtype=np.int64)
        x_idx = np.clip(((pts_w_attr[:,0]-x_min)/x_bin+x_bin/2.).astype(np.int64),0,len(arr_x)-1)
        y_idx = np.clip(((pts_w_attr[:,1]-y_min)/y_bin+y_bin/2.).astype(np.int64),0,len(arr_y)-1)

        hue_min, hue_max = dict_render[f'{hue_type}_ROI']
        hue_val = np.clip((pts_w_attr[:,self.list_attr.index(hue_type)]-hue_min)/(hue_max-hue_min),0.1,0.9)

        val_min, val_max = dict_render[f'{val_type}_ROI']
        val_val = np.clip((pts_w_attr[:,self.list_attr.index(val_type)]-val_min)/(val_max-val_min),0.5,0.9)

        xy_mesh_grid_hsv[x_idx,y_idx,0] = (hue_val*127.).astype(np.int64)
        xy_mesh_grid_hsv[x_idx,y_idx,1] = 255 # Saturation
        xy_mesh_grid_hsv[x_idx,y_idx,2] = (val_val*255.).astype(np.int64)

        xy_mesh_grid_rgb_temp = cv2.cvtColor(xy_mesh_grid_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        dilation = dict_render['DILATION']
        xy_mesh_grid_rgb_temp = cv2.dilate(xy_mesh_grid_rgb_temp, kernel=(dilation,dilation))

        xy_mesh_grid_rgb = np.full_like(xy_mesh_grid_rgb_temp, fill_value=255, dtype=np.uint8)
        x_ind_valid, y_ind_valid = np.where(np.sum(xy_mesh_grid_rgb_temp, axis=2)>0)
        xy_mesh_grid_rgb[x_ind_valid,y_ind_valid,:] = xy_mesh_grid_rgb_temp[x_ind_valid,y_ind_valid,:]

        xy_mesh_grid_rgb = np.flip(xy_mesh_grid_rgb, axis=(0,1))

        return xy_mesh_grid_rgb

    def calib_xyz(self, list_calib_xyz:list):
        arr_calib_xyz = np.array(list_calib_xyz, dtype=self.values.dtype).reshape(1,3)
        arr_calib_xyz = arr_calib_xyz.repeat(repeats=len(self.values), axis=0)
        self.values[:,:3] += arr_calib_xyz
        self.is_calibrated=True

    def roi_filter(self, dict_roi:dict):
        '''
        dict_roi
            key: 'attr', value: [attr_min, attr_max]
        e.g., {'x': [0, 100]}
        '''
        values = self.values.copy()
        for temp_key, v in dict_roi.items():
            if not (temp_key in self.list_attr):
                print(f'* {temp_key} is not in attr')
                continue
            v_min, v_max = v
            idx = self.list_attr.index(temp_key)
            values = values[
                np.where(
                    (values[:,idx]>v_min) & (values[:,idx]<v_max)
                )]
        self.values = values
        self.is_roi_filtered=True

    def render_in_o3d(self):
        o3d.visualization.draw_geometries([self._get_o3d_pcd()])

    def render_in_bev(self, dict_render:dict):
        img_bev = self._get_bev_pcd(dict_render)
        cv2.imshow('LiDAR PCD (in BEV)', img_bev)
        cv2.waitKey(0)

class RenderObject3D():
    def __init__(self, type_str:str, idx:int, cls_name:str, list_obj_info:list, conf=None):
        '''
        type
            'gt' or 'pred'
        idx
            index of object
        cls_name
            e.g., 'Sedan'
        list_obj_info
            xc, yc, zc, xl, yl, zl, rot_rad
        corners
              0 -------- 2
             /|         /|
            4 -------- 6 .
            | |        | |
            . 1 -------- 3
            |/         |/
            5 -------- 7
        '''
        # if type_str == 'gt':
        #     xc, yc, zc, rot_rad, xl, yl, zl = list_obj_info
        # else:
        xc, yc, zc, xl, yl, zl, rot_rad = list_obj_info
        
        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl, 0., xl]) / 2
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl, 0., 0.]) / 2
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl, 0., 0.]) / 2
        corners = np.row_stack((corners_x, corners_y, corners_z))
    
        rotation_matrix = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0.0],
            [np.sin(rot_rad), np.cos(rot_rad), 0.0],
            [0.0, 0.0, 1.0]])
        
        corners = rotation_matrix.dot(corners).T + np.array([[xc, yc, zc]])
        self.corners = corners[:8,:]
        self.ct_to_front = corners[8:,:]

        self.type_str = type_str
        self.idx = idx
        self.cls_name = cls_name
        self.conf = conf
        self.list_obj_info = list_obj_info

    @property
    def corners2d(self)->np.array: # 4x2
        corners = self.corners.copy()
        return np.concatenate((corners[0:1,:2], corners[2:3,:2], corners[6:7,:2], corners[4:5,:2]), axis=0)
    
    @property
    def corners_in_bbox_order(self)->np.array: # 12x3x2
        corners = self.corners.copy()
        lines_idx = [[0,0,2,1,0,2,1,3,4,4,6,5], # from
                     [1,2,3,3,4,6,5,7,5,6,7,7]] # to
        from_idx = np.array(lines_idx[0], dtype=np.int64)
        to_idx = np.array(lines_idx[1], dtype=np.int64)
        from_corners = corners[from_idx,:,np.newaxis]
        to_corners = corners[to_idx,:,np.newaxis]
            
        return np.concatenate((from_corners,to_corners), axis=-1)

class ObjectRenderer():
    def __init__(self, cfg_obj_renderer=None)->object:
        self.cfg_obj_renderer = cfg_obj_renderer
        self.list_render_obj = []
        self.list_calib_xyz = None

        if self.cfg_obj_renderer is None:
            self.dict_cls_name_to_bgr = {
                'Sedan': [0,50,255],
                'Bus or Truck': [255,0,255],
                'Motorcycle': [0,0,255],
                'Bicycle': [50,0,255],
                'Pedestrian': [255,0,0],
                'Pedestrian Group': [255,0,100],
                'Label': [0,0,0],
            }
            self.dict_cls_name_to_rgb = {
                'Sedan': [1, 0.2, 0],
                'Bus or Truck': [1, 0, 1],
                'Motorcycle': [1, 0, 0],
                'Bicycle': [1, 0, 0.2],
                'Pedestrian': [0, 0, 1],
                'Pedestrian Group': [0.4, 0, 1],
                'Label': [0, 0, 0],
            }
            self.linewidth = 2
            self.alpha = 0.5
            self.radius = 0
            self.conf_text_size = 0.8
        else:
            self.dict_cls_name_to_bgr = cfg_obj_renderer.DICT_CLS_NAME_TO_BGR
            self.dict_cls_name_to_rgb = cfg_obj_renderer.DICT_CLS_NAME_TO_RGB
            self.linewidth = cfg_obj_renderer.LINEWIDTH
            self.alpha = cfg_obj_renderer.ALPHA
            self.radius = cfg_obj_renderer.RADIUS
            self.conf_text_size = cfg_obj_renderer.CONF_TEXT_SIZE
    
    def update_calib_xyz(self, list_calib_xyz:list):
        assert len(list_calib_xyz)==3, 'components of list_calib_xyz = [x,y,z]'
        dx, dy, dz = list_calib_xyz
        self.list_calib_xyz = [dx, dy, dz]

    def update_list_render_obj_from_(self, **kwargs):
        del self.list_render_obj[:]
        self.list_render_obj = []
        pass # TODO
    
    def update_list_render_obj_from_path_label(self, path_label:str, load_as_label:bool=False):
        del self.list_render_obj[:]
        self.list_render_obj = []

        # print(self.list_calib_xyz)
        
        f = open(path_label, 'r')
        lines = f.readlines()
        # header = lines[0]
        lines = lines[1:]
        f.close()
        
        for obj_str in lines:
            list_val = obj_str.split(',')
            assert len(list_val) == 12, f'check {path_label}'
            assert list_val[0] == '*', f'check {path_label}'
            _, type_sensor, obj_idx, track_idx, cls_name,\
                        x, y, z, rot_deg, xl, yl, zl = list_val
            list_obj_info = [float(x), float(y), float(z),\
                                2.*float(xl), 2.*float(yl), 2.*float(zl), float(rot_deg)*np.pi/180.]
            if not (self.list_calib_xyz is None):
                dx, dy, dz = self.list_calib_xyz
                # print(self.list_calib_xyz)
                list_obj_info[0] += dx
                list_obj_info[1] += dy
                list_obj_info[2] += dz
            cls_name = 'Label' if load_as_label else cls_name.lstrip(' ')
            type_str = 'gt' if load_as_label else 'pred'
            self.list_render_obj.append(
                RenderObject3D(type_str, int(obj_idx), cls_name, list_obj_info)
            )
    
    def extend_list_render_obj_from_obj(self, list_render_obj:list):
        self.list_render_obj.extend(list_render_obj)

    def clear_list_render_obj(self):
        del self.list_render_obj[:]

    def render_to_img_bev(self, img_bev_in:np.array, roi_x:list, roi_y:list)->np.array:
        img_bev = img_bev_in.copy()
        x_min, x_bin, x_max = roi_x
        y_min, y_bin, y_max = roi_y
        arr_x = np.linspace(x_min, x_max-x_bin, int((x_max-x_min)/x_bin)) + x_bin/2.
        arr_y = np.linspace(y_min, y_max-y_bin, int((y_max-y_min)/y_bin)) + y_bin/2.

        len_x = len(arr_x)
        len_y = len(arr_y)
        img_bev = cv2.resize(img_bev, dsize=(len_y, len_x))

        img_bev = np.transpose(img_bev, (1,0,2)) # yx -> xy
        img_bev = np.flip(img_bev, (0,1))
        img_bev_bbox = img_bev.copy()

        for render_obj in self.list_render_obj:
            x_pts = ((render_obj.corners2d[:,0]-x_min)/x_bin+0.5).astype(int)
            y_pts = ((render_obj.corners2d[:,1]-y_min)/y_bin+0.5).astype(int)
            
            temp_bgr = self.dict_cls_name_to_bgr[render_obj.cls_name]
            img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[0],y_pts[0]), (x_pts[1],y_pts[1]), color=temp_bgr, thickness=self.linewidth)
            img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[1],y_pts[1]), (x_pts[2],y_pts[2]), color=temp_bgr, thickness=self.linewidth)
            img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[2],y_pts[2]), (x_pts[3],y_pts[3]), color=temp_bgr, thickness=self.linewidth)
            img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[3],y_pts[3]), (x_pts[0],y_pts[0]), color=temp_bgr, thickness=self.linewidth)
            
            ct_to_front = render_obj.ct_to_front[:,:2]
            ct_to_front_x = ((ct_to_front[:,0]-x_min)/x_bin+0.5).astype(int)
            ct_to_front_y = ((ct_to_front[:,1]-y_min)/y_bin+0.5).astype(int)
            img_bev_bbox = cv2.line(img_bev_bbox, (ct_to_front_x[0],ct_to_front_y[0]),\
                                    (ct_to_front_x[1],ct_to_front_y[1]), color=temp_bgr, thickness=self.linewidth)

            if self.radius != 0:
                img_bev_bbox = cv2.circle(img_bev_bbox, (ct_to_front_x[0],ct_to_front_y[0]),\
                                                        radius=self.radius, color=(0,0,0), thickness=-1)

        img_bev_bbox = self.alpha*img_bev.astype(float) + (1-self.alpha)*img_bev_bbox.astype(float)
        img_bev_bbox = np.clip(img_bev_bbox, 0., 255.).astype(np.uint8)
        img_bev_bbox = np.flip(img_bev_bbox, (0,1))
        img_bev_bbox = np.transpose(img_bev_bbox, (1,0,2)) # xy -> yx
        del img_bev
        
        return img_bev_bbox
    
    def render_to_img_perspective(self, img_perspective_in:np.array, proj:np.array):
        img_pers = img_perspective_in.copy()
        for render_obj in self.list_render_obj:
            corners_in_bbox_order = render_obj.corners_in_bbox_order.copy() # 12x3x2
            corners_in_bbox_order = np.transpose(corners_in_bbox_order, (1,0,2))
            corners_in_bbox_order = np.concatenate((corners_in_bbox_order,\
                np.array([1]).reshape((1,1,1)).repeat(12,1).repeat(2,2)), axis=0)
            corners_in_bbox_order = np.dot(proj, corners_in_bbox_order.reshape(4,24)).reshape(3,12,2)
            depth_corners = corners_in_bbox_order[2:3,:,:]
            if len(np.where(depth_corners<0)[0]) > 0: # only when all depth > 0
                continue
            corners_in_bbox_order[0:1,:,:] = corners_in_bbox_order[0:1,:,:]/depth_corners
            corners_in_bbox_order[1:2,:,:] = corners_in_bbox_order[1:2,:,:]/depth_corners
            corners_in_bbox_order = (np.round(corners_in_bbox_order)).astype(int)
            from_corners = corners_in_bbox_order[:2,:,0]
            to_corners = corners_in_bbox_order[:2,:,1]
            temp_bgr = self.dict_cls_name_to_bgr[render_obj.cls_name]
            for i in range(12):
                img_pers = cv2.line(img_pers, (from_corners[0,i], from_corners[1,i]),\
                                (to_corners[0,i], to_corners[1,i]), color=temp_bgr, thickness=self.linewidth)
            if (self.conf_text_size != 0) and (not (render_obj.conf is None)):
                conf = np.round(render_obj.conf*100., 2) # to set location of text, refer to point idx
                img_pers = cv2.putText(img_pers, f'{conf}%', (from_corners[0,-3], from_corners[1,-3]),\
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=self.conf_text_size, color=(255,255,255), thickness=1)

            # debug
            # temp_corners = render_obj.corners
            # print(f'-start---------{render_obj.idx}-----------------')
            # print(temp_corners)
            # temp_corners = np.concatenate((np.transpose(temp_corners), np.array([1]).reshape(1,1).repeat(8,1)), 0)
            # proj_corners = np.dot(proj, temp_corners)
            # proj_corners[0:1,:] = proj_corners[0:1,:]/proj_corners[2:3,:]
            # proj_corners[1:2,:] = proj_corners[1:2,:]/proj_corners[2:3,:]
            # print(proj_corners.T)
            # img_pers = cv2.putText(img_pers, f'{render_obj.idx}', (from_corners[0,-3], from_corners[1,-3]),\
            #     fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=self.conf_text_size, color=(255,255,255), thickness=1)
            # print(f'-end---------{render_obj.idx}-----------------')

        img_pers = self.alpha*img_perspective_in.astype(float) + (1-self.alpha)*img_pers.astype(float)
        img_pers = np.clip(img_pers, 0., 255.).astype(np.uint8)
            
        return img_pers

    def render_to_list_pcd(self, list_pcd:list)->list:
        lines = [[0, 1],[1, 2],[2, 3],[0, 3],[4, 5],[6, 7],[0, 4],\
                 [1, 5],[2, 6],[3, 7],[0, 2],[1, 3],[4, 6],[5, 7]]
        len_lines = len(lines)
        for render_obj in self.list_render_obj:
            color_rgb = self.dict_cls_name_to_rgb[render_obj.cls_name]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(render_obj.corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(\
                np.array(color_rgb).reshape(-1,3).repeat(len_lines,0))
            list_pcd.append(line_set)

        return list_pcd
