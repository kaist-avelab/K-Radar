import os
import os.path as osp
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

### User define ###
PATH_SAVE_TRACK = 'E:/save_tracking'
PATH_SEQ = 'E:/radar_bin_lidar_bag_files/generated_files/9'
### User define ###

class VisTrackingWithCamera():
    def __init__(self, path_seq, path_save_track):
        self.dict_cls_bgr = {
            'Sedan':[0,0,255],
            'Bus or Truck':[0,50,255],
            'Motorcycle':[0,255,0],
            'Bicycle':[0,200,255],
            'Pedestrian':[255,0,0],
            'Pedestrian Group':[255,0,100],
            'Bicycle Group':[255,100,0],
        }
        self.path_seq = path_seq
        self.path_save_track = path_save_track
        self.init_for_tracking()
        self.init_for_calibration()
    
    def init_for_tracking(self):
        self.list_unique_ids = []
        self.list_unique_center = []
        self.list_unique_timestamp = []

        self.list_unique_ids_in_screen = []
        self.list_unique_ids_l_in_screen = []
        self.list_label_ids_in_screen = []
        self.list_label_ids_l_in_screen = []

        b = np.random.choice(range(256), size=5000).reshape(-1,1)
        g = np.random.choice(range(256), size=5000).reshape(-1,1)
        r = np.random.choice(range(256), size=5000).reshape(-1,1)

        bgr = np.concatenate([b,g,r], axis=1)
        bgr = list(map(lambda x: tuple(x), bgr.tolist()))
        self.list_unique_colors = list(set(bgr))

    def init_for_calibration(self):
        img_size = (1280,720)
        dict_values = {
            'fx':557.720776478944,
            'fy':567.2136917114258,
            'px':636.720776478944,
            'py':369.3068656921387,
            'k1':-0.028873818023371287,
            'k2':0.0006023302214797655,
            'k3':0.0039573086622276855,
            'k4':-0.005047176298643093,
            'k5':0.0,
            'roll_c':0.0,
            'pitch_c':0.0,
            'yaw_c':0.0,
            'roll_l':0.0,
            'pitch_l':0.7,
            'yaw_l':-0.5,
            'x_l':0.1,
            'y_l':0.0,
            'z_l':-0.7
        }
        tr_rotation_default = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        intrinsics = np.array([
            [dict_values['fx'], 0.0, dict_values['px']],
            [0.0, dict_values['fy'], dict_values['py']],
            [0.0, 0.0, 1.0]
        ])
        distortion = np.array([
            dict_values['k1'], dict_values['k2'], dict_values['k3'], \
            dict_values['k4'], dict_values['k5']
        ]).reshape((-1,1))
        yaw_c = dict_values['yaw_c']
        pitch_c = dict_values['pitch_c']
        roll_c = dict_values['roll_c']
        r_cam = (R.from_euler('zyx', [yaw_c, pitch_c, roll_c], degrees=True)).as_matrix()
        yaw_l = dict_values['yaw_l']
        pitch_l = dict_values['pitch_l']
        roll_l = dict_values['roll_l']
        r_l = (R.from_euler('zyx', [yaw_l, pitch_l, roll_l], degrees=True)).as_matrix()
        x_l = dict_values['x_l']
        y_l = dict_values['y_l']
        z_l = dict_values['z_l']
        tr_lid_cam = np.concatenate([r_l, np.array([x_l,y_l,z_l]).reshape(-1,1)], axis=1)
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics[j,i] = ncm[j, i]
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, r_cam, ncm, img_size, cv2.CV_32FC1)
        P2 = np.insert(intrinsics, 3, values=[0,0,0], axis=1) # K = intrinsics
        R0_hom = r_cam
        R0_hom = np.insert(R0_hom,3,values=[0,0,0],axis=0)
        R0_hom = np.insert(R0_hom,3,values=[0,0,0,1],axis=1) # 4x4
        LidarToCamera = np.insert(tr_lid_cam, 3, values=[0,0,0,1], axis=0)
        self.proj = np.matmul(np.matmul(P2,R0_hom),np.matmul(LidarToCamera, tr_rotation_default))
        self.idx_drawing_line_0 = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
        self.idx_drawing_line_1 = [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]
        # xl=0,yl=0,zl=0
        # corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
        # corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
        # corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 
        #                       0   1    2    3   4   5    6   7
    def get_tuple_object(self, line):
        list_values = line.split(',')

        if list_values[0] != '*':
            return None

        list_values = list_values[1:]

        idx_now = int(list_values[0])
        idx_obj = int(list_values[1])
        cls_name = list_values[2][1:]

        x = float(list_values[3])
        y = float(list_values[4])
        z = float(list_values[5])
        theta = float(list_values[6])
        theta = theta*np.pi/180.
        l = 2*float(list_values[7])
        w = 2*float(list_values[8])
        h = 2*float(list_values[9])

        return (cls_name, [x,y,z,theta,l,w,h], idx_obj, idx_now) # idx_obj: tracking id

    def get_corresponding_pix_from_object_tuple(self, obj_tuple, \
            vis_range=100, pix_height=800, pix_width=1280):
        cls_name, bbox_params, idx_obj, idx_now = obj_tuple
        x, y, z, rot_rad, xl, yl, zl = bbox_params
        ### deriving four points ###
        corners_x = np.array([xl, xl, -xl, -xl, 0, xl, xl,      xl+5.]) / 2 
        corners_y = np.array([yl, -yl, -yl, yl, 0, 0,  -yl*1.5, -yl*1.5]) / 2 
        corners = np.row_stack((corners_x, corners_y))
        rot_matrix = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad)],
            [np.sin(rot_rad), np.cos(rot_rad)]
        ])
        corners = rot_matrix.dot(corners) + np.array([[x], [y]])
        ### deriving four points ###

        ### change into pixels ###
        pix_per_m = float(pix_height)/vis_range
        min_meter_x = pix_width/pix_per_m/2.
        corners[1,:] = min_meter_x - corners[1,:] # pixel x
        corners = corners*pix_per_m
        corners[0,:] = pix_height - corners[0,:] # pixel y
        ### change into pixels ###

        corners = np.around(corners).astype(int)

        ### corners point cloud ###
        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 

        corners_3d = np.row_stack((corners_x, corners_y, corners_z))
    
        rotation_matrix = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0.0],
            [np.sin(rot_rad), np.cos(rot_rad), 0.0],
            [0.0, 0.0, 1.0]])
        
        corners_3d = rotation_matrix.dot(corners_3d).T + np.array([[x, y, z]])
        corners_3d = np.insert(np.transpose(corners_3d, (1,0)), 3 , 1, axis=0)
        cam_3d = np.matmul(self.proj, corners_3d)
        cam_3d[:2] /= cam_3d[2,:]
        cam_3d = np.around(cam_3d).astype(int)
        ### corners point cloud ###
        
        return corners, cls_name, idx_obj, idx_now, [x, y], cam_3d        

    def dict_file_paths_from_label_txt(self, label_txt, path_attr):
        temp_dict = dict()
        f = open(osp.join(path_attr, 'info_label', label_txt), 'r')
        lines = f.readlines()
        header = lines[0]
        indices = ((header.split(',')[0]).split('=')[1]).split('_')
        timestamp = float(((header.split(',')[1]).split('=')[1]).rstrip('\n'))
        idx_rdr, idx_os64, idx_cam_front, idx_os128, idx_cam_lrr = indices
        temp_dict['path_img'] = osp.join(path_attr, 'cam-front', f'cam-front_{idx_cam_front}.png')
        temp_dict['path_pcd'] = osp.join(path_attr, 'os2-64', f'os2-64_{idx_os64}.pcd')
        temp_dict['path_bev_50'] = osp.join(path_attr, 'lidar_bev_image', f'lidar_bev_50_{idx_os64}.png')
        temp_dict['path_bev_100'] = osp.join(path_attr, 'lidar_bev_image', f'lidar_bev_100_{idx_os64}.png')
        temp_dict['path_bev_110'] = osp.join(path_attr, 'lidar_bev_image', f'lidar_bev_110_{idx_os64}.png')
        temp_dict['timestamp'] = timestamp
        ### Read labels ###
        lines = lines[1:]

        list_objects = []
        for line in lines:
            corner_pixels, cls_name, idx_obj, idx_now, center, cam_3d = \
                self.get_corresponding_pix_from_object_tuple(\
                self.get_tuple_object(line))
            list_objects.append([corner_pixels, cls_name, idx_obj, idx_now, center, cam_3d])
        f.close()
        temp_dict['objects'] = list_objects

        return temp_dict

    def get_label_paths_from_path_seq(self, max_frame=1000):
        list_labels = os.listdir(osp.join(self.path_seq, 'info_label'))
        self.list_dict_items = []
        for idx_frame, path_label in enumerate(list_labels):
            print(path_label)
            if idx_frame > max_frame:
                break
            temp_dict = self.dict_file_paths_from_label_txt(path_label, self.path_seq)
            ### Update tracking info ###
            list_unique_id = []
            list_velocity = []
            list_bgr = []
            for temp_object in temp_dict['objects']:
                corner_pixels, cls_name, idx_obj, idx_now, center, cam_3d = temp_object

                if idx_obj == -1:
                    unique_id = len(self.list_unique_ids)
                    self.list_unique_ids.append(unique_id)
                    self.list_unique_timestamp.append(temp_dict['timestamp'])
                    self.list_unique_center.append(center)

                    self.list_unique_ids_in_screen.append(unique_id)
                    self.list_label_ids_in_screen.append(idx_now)

                    bgr = self.list_unique_colors[unique_id]
                    velocity = -1
                else:
                    # try:
                    # find idx_obj is in last ids_l
                    corr_idx = self.list_label_ids_l_in_screen.index(idx_obj)
                    unique_id = self.list_unique_ids_l_in_screen[corr_idx]

                    last_timestamp = self.list_unique_timestamp[unique_id]
                    last_center = self.list_unique_center[unique_id]
                    now_timestamp = temp_dict['timestamp']
                    now_center = center
                    bgr = self.list_unique_colors[unique_id]
                    velocity = np.sqrt((now_center[0]-last_center[0])**2+(now_center[1]-last_center[1])**2)/(now_timestamp-last_timestamp)

                    # km/h
                    velocity = np.around(velocity*3.6, 2)

                    # sign
                    l2_last = (last_center[0])**2 + (last_center[1])**2
                    l2_now = (now_center[0])**2 + (now_center[1])**2
                    if l2_last > l2_now:
                        velocity = -velocity

                    self.list_unique_ids_in_screen.append(unique_id)
                    self.list_label_ids_in_screen.append(idx_now)
                    self.list_unique_center[unique_id] = now_center
                    self.list_unique_timestamp[unique_id] = now_timestamp
                    # except:
                    #     # continue
                    #     unique_id = len(self.list_unique_ids)
                    #     self.list_unique_ids.append(unique_id)
                    #     self.list_unique_timestamp.append(temp_dict['timestamp'])
                    #     self.list_unique_center.append(center)

                    #     self.list_unique_ids_in_screen.append(unique_id)
                    #     self.list_label_ids_in_screen.append(idx_now)

                    #     bgr = self.list_unique_colors[unique_id]
                    #     velocity = -1
                
                list_unique_id.append(unique_id)
                list_bgr.append(bgr)
                list_velocity.append(velocity)
            
            self.list_label_ids_l_in_screen = (self.list_label_ids_in_screen).copy()
            self.list_label_ids_in_screen = []
            self.list_unique_ids_l_in_screen = (self.list_unique_ids_in_screen).copy()
            self.list_unique_ids_in_screen = []
            temp_dict['objects_id'] = list_unique_id
            temp_dict['objects_bgr'] = list_bgr
            temp_dict['objects_velocity'] = list_velocity

            ### Update tracking info ###
            self.list_dict_items.append(temp_dict)

    def process_one_sample_for_vis(self, idx, vis_range=100, duration=1000, is_save=False):
        temp_dict = self.list_dict_items[idx]
        if idx == 0:
            f = open(f'{self.path_save_track}/debug.txt', 'w')
        else:
            f = open(f'{self.path_save_track}/debug.txt', 'a')
        file_name = temp_dict['path_pcd']
        f.write(f'{idx}: {file_name}\n')
        f.close()

        vis_key = f'path_bev_{vis_range}'
        vis_img = cv2.imread(temp_dict[vis_key])

        vis_cam = cv2.imread(temp_dict['path_img'])[:,:1280].copy()
        vis_cam = cv2.remap(vis_cam, self.map_x, self.map_y, cv2.INTER_LINEAR)
        
        unique_id = temp_dict['objects_id']
        bgr = temp_dict['objects_bgr']
        velocity = temp_dict['objects_velocity']
        for idx_id, temp_bbox in enumerate(temp_dict['objects']):
            corner_pixels, cls_name, idx_obj, idx_now, center, cam_3d = temp_bbox
            bgr_cl = bgr[idx_id]

            ### Vis for bev ###
            for idx_corner in range(3):
                vis_img = cv2.line(vis_img, \
                    (corner_pixels[1,idx_corner], corner_pixels[0,idx_corner]), \
                    (corner_pixels[1,idx_corner+1], corner_pixels[0,idx_corner+1]), \
                    bgr_cl, thickness=2)
            
            vis_img = cv2.line(vis_img, \
                (corner_pixels[1,3], corner_pixels[0,3]), \
                (corner_pixels[1,0], corner_pixels[0,0]), \
                bgr_cl, thickness=2)

            vis_img = cv2.arrowedLine(vis_img, \
                (corner_pixels[1,4], corner_pixels[0,4]), \
                (corner_pixels[1,5], corner_pixels[0,5]), \
                bgr_cl, thickness=2, tipLength=0.3)
            
            cv2.putText(vis_img, f'{unique_id[idx_id]}', \
                (corner_pixels[1,6], corner_pixels[0,6]), \
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0))

            if velocity[idx_id] != -1:
                cv2.putText(vis_img, f'{velocity[idx_id]} [km/h]', \
                    (corner_pixels[1,7], corner_pixels[0,7]), \
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0))
            ### Vis for bev ###

            ### Vis for cam ###
            #### Check sum of valid pixels > 4 ###
            checker_valid = (cam_3d[0,:]<1280) & (cam_3d[0,:]>=0) & (cam_3d[1,:]<720) & (cam_3d[1,:]>=0)
            num_valid = np.sum(np.array(checker_valid))
            if num_valid < 8:
                continue
            
            # if np.sum(np.array(np.array(corner_pixels[0,:] < 5))) > 0:
            #     continue

            for idx_corner in range(12):
                vis_cam = cv2.line(vis_cam, \
                    (cam_3d[0,self.idx_drawing_line_0[idx_corner]], cam_3d[1,self.idx_drawing_line_0[idx_corner]]), \
                    (cam_3d[0,self.idx_drawing_line_1[idx_corner]], cam_3d[1,self.idx_drawing_line_1[idx_corner]]), \
                    bgr_cl, thickness=2)
                vis_cam = cv2.putText(vis_cam, f'{unique_id[idx_id]}', \
                    (cam_3d[0,0], cam_3d[1,0]),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, bgr_cl)
            ### Vis for cam ###
            
        cv2.imshow('img', vis_img)
        cv2.imshow('cam', vis_cam)
        if is_save:
            cv2.imwrite(f'{self.path_save_track}/img_bev_{idx}.png', vis_img)
            cv2.imwrite(f'{self.path_save_track}/img_cam_{idx}.png', vis_cam)
        cv2.waitKey(duration)

    def video_maker(self, max_frame):
        video = cv2.VideoWriter(f'{self.path_save_track}/video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (1280, 800))
        for idx in tqdm(range(max_frame)):
            vis_bev = cv2.imread(f'{self.path_save_track}/img_bev_{idx}.png')
            vis_cam = cv2.imread(f'{self.path_save_track}/img_cam_{idx}.png')
            vis_cam = cv2.resize(vis_cam, (480, 270), interpolation=cv2.INTER_LINEAR)
            vis_bev[:270,800:,:] = vis_cam
            video.write(vis_bev)
        video.release()

if __name__ == '__main__':
    MAX_FRAME = 287
    vis_tracking = VisTrackingWithCamera(PATH_SEQ, PATH_SAVE_TRACK)
    vis_tracking.get_label_paths_from_path_seq(max_frame=MAX_FRAME)
    for i in tqdm(range(MAX_FRAME)):
        vis_tracking.process_one_sample_for_vis(i, duration=100, is_save=True)
    vis_tracking.video_maker(MAX_FRAME)
