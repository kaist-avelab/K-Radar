'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import os
import os.path as osp
import numpy as np
import cv2

from easydict import EasyDict

### Configuration ###
DICT_CFG = {
    'GENERAL': {
        'LOAD_TYPE': 'label', # in ['label', 'inf']
    },
    'DIR': {
        'LIST_DIR': ['/media/donghee/HDD_3/K-Radar/radar_bin_lidar_bag_files/generated_files'],
        'DIR_PROCESSED_CAM': '/media/donghee/HDD_3/K-Radar/dir_processed_img/img_driving_corridor',
        'DIR_RENDERED_LDR': '/media/donghee/HDD_3/K-Radar/dir_rendered_lpc',
        'DIR_RENDERED_RDR': '/media/donghee/HDD_3/KRadar_radar_frames',
        'DIR_INF': '/media/donghee/HDD_3/K-Radar_inf_results/driving_corridor_only_sedan/RTNHPP/conf_0_55',
        'DIR_FILT_SRT': '/media/donghee/HDD_3/K-Radar/dir_sp_filtered', # RC_FUSION, RTNHPP
        'SEQ_LOAD': ['55'], # None: Loading all seqs
    },
    'RENDER': { # After resize
        'LDR_XY': [0, 0.1, 80, -40, 0.1, 40], # min, bin, max in XY
        'RDR_XY': [0, 0.1, 80, -40, 0.1, 40], # min, bin, max in XY
        'LINEWIDTH': 2,
        'ALPHA': 0.5,
        'RADIUS': 0,
        'CONF_TEXT_SIZE': 0.8,
        'LABEL_BGR': [0,50,255],
        'DICT_CLS_NAME_TO_BGR': {
            'Sedan': [0,255,0],
        },
        'SRT': {
            'BGR': [0,0,0],
            'DILATION': 5,
            'ALPHA': 0.5,
        },
    },
}
### Configuration ###

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

class RenderInferResults():
    def __init__(self, cfg:dict=None)->object:
        self.cfg = cfg
        list_dict_item = []
        if cfg.DIR.SEQ_LOAD is None:
            list_seqs = os.listdir(cfg.DIR.PATH_INF)
        else:
            list_seqs = cfg.DIR.SEQ_LOAD
        self.list_seqs = list_seqs
        dir_inf = cfg.DIR.DIR_INF
        self.dir_rdr = cfg.DIR.DIR_RENDERED_RDR
        self.dir_ldr = cfg.DIR.DIR_RENDERED_LDR
        self.dir_camf = cfg.DIR.DIR_PROCESSED_CAM
        self.dir_srt = cfg.DIR.DIR_FILT_SRT
        for seq_name in list_seqs:
            list_frames = sorted(os.listdir(osp.join(dir_inf, seq_name)))
            for file_name in list_frames:
                path_inf = osp.join(dir_inf, seq_name, file_name)
                list_dict_item.append(self._load_inf_info(path_inf))
        self.list_dict_item = list_dict_item
        
        self.proj = np.load(osp.join(self.dir_camf, 'proj.npy'))
    
    def __repr__(self)->str:
        print(f'total {len(self.list_path_inf)} frames')
        str_seqs = '1-58' if len(self.list_seqs) == 58 else f'{self.list_seqs}'
        print(f'loaded seqs = {str_seqs}')

    def __getitem__(self, idx:int)->dict:
        return self.list_dict_item[idx]
    
    def __len__(self)->int:
        return len(self.list_dict_item)

    def _load_inf_info(self, path_inf)->dict:
        f = open(path_inf)
        lines = f.readlines()
        f.close()
        lines = list(map(lambda x: x.rstrip('\n'), lines))
        header = lines[0]
        objs = lines[2:]
        list_gt_render_obj = []
        list_pred_render_obj = []
        is_gt = True
        for temp_obj in objs:
            if temp_obj == '-':
                is_gt = False
            else:
                list_vals = temp_obj.split(',')
                idx = int(list_vals[0])
                cls_name = list_vals[1]
                list_obj_info = list(map(lambda x: float(x), list_vals[2:9]))
                if is_gt:
                    type_str = 'gt'
                    render_obj = RenderObject3D(type_str, idx, cls_name, list_obj_info)
                    list_gt_render_obj.append(render_obj)
                else:
                    type_str = 'pred'
                    render_obj = RenderObject3D(type_str, idx, cls_name, list_obj_info, float(list_vals[9]))
                    list_pred_render_obj.append(render_obj)
        idx_str, tstamp_str = header.split(',')
        idx_rdr, idx_ldr, idx_camf, _, _ = idx_str.split('=')[1].split('_')
        tstamp = float(tstamp_str.split('=')[1])
        seq_name = path_inf.split('/')[-2]

        dict_path = {
            'inf': path_inf,
            'camf': osp.join(self.dir_camf, seq_name, f'front_{idx_camf}.npy'),
            'rdr_bev': osp.join(self.dir_rdr, seq_name, f'tesseract_{idx_rdr}.png'),
            'ldr_bev': osp.join(self.dir_ldr, seq_name, f'os2-64_{idx_ldr}.png'),
            'rdr_srt': osp.join(self.dir_srt, 'temp', seq_name, f'rp_{idx_rdr}.npy')
        }
        dict_item = {
            'path': dict_path,
            'tstamp': tstamp,
            'pred': list_pred_render_obj,
            'gt': list_gt_render_obj,
        }

        return dict_item
    
    def _get_rendered_img(self, dict_item:dict,
                type_sensor:str='rdr', is_w_label:bool=True, is_w_pred:bool=True)->np.array:
        '''
        type_sensor
            in ['rdr', 'ldr', 'camf']
        '''
        dict_cls_bgr = self.cfg.RENDER.DICT_CLS_NAME_TO_BGR
        label_bgr = self.cfg.RENDER.LABEL_BGR
        linewidth = self.cfg.RENDER.LINEWIDTH
        alpha = self.cfg.RENDER.ALPHA
        radius = self.cfg.RENDER.RADIUS
        conf_text_size = self.cfg.RENDER.CONF_TEXT_SIZE

        if type_sensor in ['rdr', 'ldr']:
            img_bev = cv2.imread(dict_item['path'][type_sensor+'_bev'])
            if type_sensor == 'rdr':
                img_bev = np.flip(img_bev, axis=1)

            x_min, x_bin, x_max, y_min, y_bin, y_max =\
                self.cfg.RENDER.get(f'{type_sensor.upper()}_XY', None)
            arr_x = np.linspace(x_min, x_max-x_bin, int((x_max-x_min)/x_bin)) + x_bin/2.
            arr_y = np.linspace(y_min, y_max-y_bin, int((y_max-y_min)/y_bin)) + y_bin/2.
            len_x = len(arr_x)
            len_y = len(arr_y)
            img_bev = cv2.resize(img_bev, dsize=(len_y, len_x))

            if is_w_label or is_w_pred:
                img_bev = np.transpose(img_bev, (1,0,2)) # yx -> xy
                img_bev = np.flip(img_bev, (0,1))
                img_bev_bbox = img_bev.copy()

                list_gt_render_obj = dict_item['gt']
                list_pred_render_obj = dict_item['pred']
                if is_w_label:
                    for render_obj in list_gt_render_obj:
                        x_pts = ((render_obj.corners2d[:,0]-x_min)/x_bin+0.5).astype(int)
                        y_pts = ((render_obj.corners2d[:,1]-y_min)/y_bin+0.5).astype(int)
                        
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[0],y_pts[0]), (x_pts[1],y_pts[1]), color=label_bgr, thickness=linewidth)
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[1],y_pts[1]), (x_pts[2],y_pts[2]), color=label_bgr, thickness=linewidth)
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[2],y_pts[2]), (x_pts[3],y_pts[3]), color=label_bgr, thickness=linewidth)
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[3],y_pts[3]), (x_pts[0],y_pts[0]), color=label_bgr, thickness=linewidth)
                        
                        ct_to_front = render_obj.ct_to_front[:,:2]
                        ct_to_front_x = ((ct_to_front[:,0]-x_min)/x_bin+0.5).astype(int)
                        ct_to_front_y = ((ct_to_front[:,1]-y_min)/y_bin+0.5).astype(int)
                        img_bev_bbox = cv2.line(img_bev_bbox, (ct_to_front_x[0],ct_to_front_y[0]),\
                                                (ct_to_front_x[1],ct_to_front_y[1]), color=label_bgr, thickness=linewidth)

                        if radius != 0:
                            img_bev_bbox = cv2.circle(img_bev_bbox, (ct_to_front_x[0],ct_to_front_y[0]),\
                                                                    radius=radius, color=(0,0,0), thickness=-1)
                if is_w_pred:
                    for render_obj in list_pred_render_obj:
                        obj_bgr = dict_cls_bgr[render_obj.cls_name]
                        x_pts = ((render_obj.corners2d[:,0]-x_min)/x_bin+0.5).astype(int)
                        y_pts = ((render_obj.corners2d[:,1]-y_min)/y_bin+0.5).astype(int)
                        
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[0],y_pts[0]), (x_pts[1],y_pts[1]), color=obj_bgr, thickness=linewidth)
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[1],y_pts[1]), (x_pts[2],y_pts[2]), color=obj_bgr, thickness=linewidth)
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[2],y_pts[2]), (x_pts[3],y_pts[3]), color=obj_bgr, thickness=linewidth)
                        img_bev_bbox = cv2.line(img_bev_bbox, (x_pts[3],y_pts[3]), (x_pts[0],y_pts[0]), color=obj_bgr, thickness=linewidth)

                        ct_to_front = render_obj.ct_to_front[:,:2]
                        ct_to_front_x = ((ct_to_front[:,0]-x_min)/x_bin+0.5).astype(int)
                        ct_to_front_y = ((ct_to_front[:,1]-y_min)/y_bin+0.5).astype(int)
                        img_bev_bbox = cv2.line(img_bev_bbox, (ct_to_front_x[0],ct_to_front_y[0]),\
                                                (ct_to_front_x[1],ct_to_front_y[1]), color=obj_bgr, thickness=linewidth)
                
                img_bev_bbox = alpha*img_bev.astype(float) + (1-alpha)*img_bev_bbox.astype(float)
                img_bev_bbox = np.clip(img_bev_bbox, 0., 255.).astype(np.uint8)
                img = img_bev_bbox
            else:
                img = img_bev
            
            img = np.flip(img, (0,1))
            img = np.transpose(img, (1,0,2)) # xy -> yx
        elif type_sensor == 'camf':
            img_camf = np.load(dict_item['path']['camf'])

            if is_w_label or is_w_pred:
                img_camf_bbox = img_camf.copy()

                list_gt_render_obj = dict_item['gt']
                list_pred_render_obj = dict_item['pred']
                if is_w_label:
                    for render_obj in list_gt_render_obj:
                        corners_in_bbox_order = render_obj.corners_in_bbox_order.copy() # 12x3x2
                        corners_in_bbox_order = np.transpose(corners_in_bbox_order, (1,0,2))
                        corners_in_bbox_order = np.concatenate((corners_in_bbox_order,\
                            np.array([1]).reshape((1,1,1)).repeat(12,1).repeat(2,2)), axis=0)
                        corners_in_bbox_order = np.dot(self.proj, corners_in_bbox_order.reshape(4,24)).reshape(3,12,2)
                        depth_corners = corners_in_bbox_order[2:3,:,:]
                        if len(np.where(depth_corners<0)[0]) > 0: # only when all depth > 0
                            continue
                        corners_in_bbox_order[0:1,:,:] = corners_in_bbox_order[0:1,:,:]/depth_corners
                        corners_in_bbox_order[1:2,:,:] = corners_in_bbox_order[1:2,:,:]/depth_corners
                        corners_in_bbox_order = (np.round(corners_in_bbox_order)).astype(int)
                        from_corners = corners_in_bbox_order[:2,:,0]
                        to_corners = corners_in_bbox_order[:2,:,1]

                        for i in range(12):
                            img_camf_bbox = cv2.line(img_camf_bbox, (from_corners[0,i], from_corners[1,i]),\
                                                    (to_corners[0,i], to_corners[1,i]), color=label_bgr, thickness=linewidth)
                if is_w_pred:
                    for render_obj in list_pred_render_obj:
                        obj_bgr = dict_cls_bgr[render_obj.cls_name]
                        corners_in_bbox_order = render_obj.corners_in_bbox_order.copy() # 12x3x2
                        corners_in_bbox_order = np.transpose(corners_in_bbox_order, (1,0,2))
                        corners_in_bbox_order = np.concatenate((corners_in_bbox_order,\
                            np.array([1]).reshape((1,1,1)).repeat(12,1).repeat(2,2)), axis=0)
                        corners_in_bbox_order = np.dot(self.proj, corners_in_bbox_order.reshape(4,24)).reshape(3,12,2)
                        depth_corners = corners_in_bbox_order[2:3,:,:]
                        if len(np.where(depth_corners<0)[0]) > 0: # only when all depth > 0
                            continue
                        corners_in_bbox_order[0:1,:,:] = corners_in_bbox_order[0:1,:,:]/depth_corners
                        corners_in_bbox_order[1:2,:,:] = corners_in_bbox_order[1:2,:,:]/depth_corners
                        corners_in_bbox_order = (np.round(corners_in_bbox_order)).astype(int)
                        from_corners = corners_in_bbox_order[:2,:,0]
                        to_corners = corners_in_bbox_order[:2,:,1]

                        for i in range(12):
                            img_camf_bbox = cv2.line(img_camf_bbox, (from_corners[0,i], from_corners[1,i]),\
                                                    (to_corners[0,i], to_corners[1,i]), color=obj_bgr, thickness=linewidth)
                        
                        if conf_text_size != 0:
                            conf = np.round(render_obj.conf*100., 2) # to set location of text, refer to point idx
                            img_camf_bbox = cv2.putText(img_camf_bbox, f'{conf}%', (from_corners[0,-3], from_corners[1,-3]),\
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=conf_text_size, color=(255,255,255), thickness=1)
                
                img_camf_bbox = alpha*img_camf.astype(float) + (1-alpha)*img_camf_bbox.astype(float)
                img_camf_bbox = np.clip(img_camf_bbox, 0., 255.).astype(np.uint8)
                img = img_camf_bbox
            else:
                img = img_camf

        return img

    def _get_rendered_img_w_srt(self, dict_item:dict,
                type_sensor:str='rdr', is_w_label:bool=True, is_w_pred:bool=True)->np.array:
        '''
        type_sensor
            in ['rdr', 'ldr']
        '''
        img_bev = self._get_rendered_img(dict_item, type_sensor, is_w_label, is_w_pred)

        x_min, x_bin, x_max, y_min, y_bin, y_max =\
            self.cfg.RENDER.get(f'{type_sensor.upper()}_XY', None)
        arr_x = np.linspace(x_min, x_max-x_bin, int((x_max-x_min)/x_bin)) + x_bin/2.
        arr_y = np.linspace(y_min, y_max-y_bin, int((y_max-y_min)/y_bin)) + y_bin/2.

        path_rdr_srt = dict_item['path']['rdr_srt']
        path_split = path_rdr_srt.split('/')
        if 'type_srt' in list(dict_item.keys()):
            path_split[-3] = dict_item['type_srt']
            path_rdr_srt = '/'.join(path_split)
            rdr_srt = np.load(path_rdr_srt) # N, 6 (X, Y, Z, PW, DOPPLER)
        else:
            print('set type of rdr srt first of dict e.g., dict_item[''type_srt'']=''filtered_005_5E''')
            return img_bev

        # img_bev = np.transpose(img_bev, (1,0,2)) # yx -> xy
        img_bev = np.flip(img_bev, (0,1))

        img_rdr_srt = np.zeros_like(img_bev, dtype=float)
        rdr_srt = rdr_srt[np.where(
            (rdr_srt[:,0]>x_min) & (rdr_srt[:,0]<x_max) &
            (rdr_srt[:,1]>y_min) & (rdr_srt[:,1]<y_max)
        )] # ROI fitering
        rdr_srt = np.array(sorted(rdr_srt.tolist(), key=lambda x: x[4])) # sorting along power
        ind_x = np.clip(((rdr_srt[:,0]-x_min)/x_bin+x_bin/2.).astype(np.int64), 0, len(arr_x)-1)
        ind_y = np.clip(((rdr_srt[:,1]-y_min)/y_bin+y_bin/2.).astype(np.int64), 0, len(arr_y)-1)
        rdr_srt_pw = np.log10(rdr_srt[:,3:4])
        value_normalized_along_pw = ((rdr_srt_pw-np.min(rdr_srt_pw))/(np.max(rdr_srt_pw)-np.min(rdr_srt_pw))*255.).astype(np.uint8)
        img_rdr_srt[ind_x,ind_y,:] = value_normalized_along_pw.repeat(3, axis=1)

        dilation = self.cfg.RENDER.SRT.DILATION
        
        cv2.imshow('img_bev', img_bev)
        cv2.imshow('temp', img_rdr_srt)

        img_rdr_srt = cv2.dilate(img_rdr_srt, (dilation, dilation))

        cv2.imshow('dilated', img_rdr_srt)

        ind_x_none_zero, ind_y_none_zero = np.where(np.sum(img_rdr_srt, axis=2)!=0)

        img_bev_copied = img_bev.copy()
        img_bev_copied[ind_x_none_zero, ind_y_none_zero, :] = img_rdr_srt[ind_x_none_zero, ind_y_none_zero, :]

        cv2.imshow('img_bev_copied', img_bev_copied)

        cv2.waitKey(0)

        alpha = self.cfg.RENDER.SRT.ALPHA
        img_bev = alpha*img_bev.astype(float) + (1-alpha)*img_bev_copied.astype(float)

        img_bev = np.flip(img_bev, (0,1))
        # img_bev = np.transpose(img_bev, (1,0,2)) # xy -> yx

        return img_bev
        
    def show_rendered_img(self, dict_item:dict,
                type_sensor:str='rdr', is_w_label:bool=True, is_w_pred:bool=True, save_dir:str=None):
        '''
        type_sensor
            in ['rdr', 'ldr', 'camf', 'all']
        '''
        if type_sensor != 'all':
            cv2.imshow(type_sensor, self._get_rendered_img(dict_item,
                                    type_sensor, is_w_label, is_w_pred))
            cv2.waitKey(0)
        else:
            img_camf = self._get_rendered_img(dict_item, 'camf', is_w_label, is_w_pred)
            img_ldr = self._get_rendered_img(dict_item, 'ldr', is_w_label, is_w_pred)
            img_rdr = self._get_rendered_img(dict_item, 'rdr', is_w_label, is_w_pred)

            # cv2.imshow('temp', img_camf[:, 200:-200,:])
            
            h_cam, w_cam, _ = img_camf.shape
            h_ldr, w_ldr, _ = img_ldr.shape
            h_rdr, w_rdr, _ = img_rdr.shape

            refer_h_int = min(h_cam, h_ldr, h_rdr)
            refer_h = float(refer_h_int)
            img_camf = cv2.resize(img_camf, (int(w_cam*refer_h/h_cam), refer_h_int))
            img_ldr = cv2.resize(img_ldr, (int(w_ldr*refer_h/h_ldr), refer_h_int))
            img_rdr = cv2.resize(img_rdr, (int(w_rdr*refer_h/h_rdr), refer_h_int))

            img_concat = cv2.hconcat([img_camf,img_ldr,img_rdr])
            cv2.imshow(type_sensor, img_concat)

            # cv2.imshow('camf', img_camf)
            # cv2.imshow('ldr', img_ldr)
            # cv2.imshow('rdr', img_rdr)

            if save_dir is None:
                cv2.waitKey(0) # 50)
            else:
                path_inf = dict_item['path']['inf']
                list_split = path_inf.split('/')
                model_name = list_split[-4]
                conf_name = list_split[-3]
                seq_name = list_split[-2]
                data_name = list_split[-1].split('.')[0] + '.png'
                path_save_dir = osp.join(save_dir, seq_name, model_name, conf_name)

                os.makedirs(path_save_dir, exist_ok=True)
                cv2.imwrite(osp.join(path_save_dir, data_name), img_concat)

    def show_redered_img_w_srt(self, dict_item:dict,
                type_sensor:str='rdr', is_w_label:bool=True, is_w_pred:bool=True):
        '''
        type_sensor
            in ['rdr', 'ldr', 'camf', 'all']
        '''
        cv2.imshow(type_sensor, self._get_rendered_img_w_srt(dict_item,
                                    type_sensor, is_w_label, is_w_pred))
        cv2.waitKey(0)

if __name__ == '__main__':
    '''
    * rotation for Radar is not required, since high power is located on wheel of vehicles
    '''
    from tqdm import tqdm
    render_infer_results = RenderInferResults(cfg=EasyDict(DICT_CFG))
    for i in tqdm(range(len(render_infer_results))):
        render_infer_results.show_rendered_img(render_infer_results[i], 'all', is_w_label=True, is_w_pred=True, save_dir='/media/donghee/HDD_3/Temp_save_dir')

    # render_infer_results[0]['type_srt'] = 'filtered_005_10E'
    # render_infer_results.show_redered_img_w_srt(render_infer_results[0], 'rdr', True, True)
