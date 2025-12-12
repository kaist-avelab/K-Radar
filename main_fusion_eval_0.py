import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'

import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from tqdm import tqdm
from scipy.io import loadmat

try:
    from utils.util_config import cfg, cfg_from_yaml_file
    from utils.util_pipeline import *
    from tools.get_multi_modal_features.feature_utils import *
    from utils.kitti_eval.eval_revised import get_official_eval_result_revised
    import utils.kitti_eval.kitti_common as kitti
except:
    import sys
    import os.path as osp
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from utils.util_config import cfg, cfg_from_yaml_file
    from utils.util_pipeline import *
    from tools.get_multi_modal_features.feature_utils import *
    from utils.kitti_eval.eval_revised import get_official_eval_result_revised
    import utils.kitti_eval.kitti_common as kitti

### [Configurations] ###
# We note # [TBC:] in the configurations to be changed.
PATH_CFG = './configs/A2F_v2_0_final.yml'
PATH_PT = './pretrained/A2F_v2_0_final_10.pt'
### [Configurations] ###

class ResultVis():
    def __init__(self, path_cfg=None, path_pt=None):
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)

        ### [Custom Configurations] ###
        self.cfg.DATASET.NAME = 'KRadarFusion_v1_0_AffectedSensor'
        self.cfg.DATASET.portion = None # ['10'] # [TBC] Change the portion to the desired seqs.
        ### [Custom Configurations] ###
        
        ### [Config for front cam] ###
        self.cfg.DATASET['t_params'] = dict()
        self.cfg.DATASET.t_params.load = True
        self.cfg.DATASET.t_params.dir = './resources/cam_calib/T_params_seq'
        self.cfg.DATASET.t_params.ref_sensor = 'radar'
        ### [Config for front cam] ###

        ### [Colors & Details] ###
        self.colors_pred  = [1., 1., 0.] # 0.6 for cam
        self.colors_label = [0.6, 0., 0.]
        self.show_details = False # arrow & conf score
        ### [Colors & Details] ###

        ### [Config for SNN] ###
        # """
        # 1. portion=['10'], indices=np.arange(60, 150)  (split='test'), 62
        # 2. portion=['48'], indices=np.arange(170, 180) (split='test'), 174
        # 3. portion=['57'], indices=np.arange(178, 188) (split='test'), 179
        # """
        # if self.cfg.MODEL.NAME == 'SpikingRTNH':
        #     print('* Change configuration for SNN')
        #     self.cfg.MODEL.SPIKING.STEP = 1
        #     self.cfg.MODEL.SPIKING.GRADUAL_SPIKING.IS_GRADUAL = True
        #     self.cfg.MODEL.SPIKING.GRADUAL_SPIKING.PORTION = 0.8
        ### [Config for SNN] ###

        set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
        self.network = build_network(self).cuda()
        
        if path_pt is not None:
            self.network.load_state_dict(torch.load(path_pt))

        consider_effects = False
        if not consider_effects:
            self.cfg_effect = None
        else:
            self.cfg_effect = dict(
                cam = dict(
                    path_ori = dict(
                        crack = './resources/imgs/img_crack.png',
                        light_scratch = './resources/imgs/img_light_scratch.png',
                        severe_scratch = './resources/imgs/img_severe_scratch.png',
                    ),
                    path_cropped = dict(
                        crack = './resources/imgs/img_crack_cropped.png',
                        light_scratch = './resources/imgs/img_light_scratch_cropped.png',
                        severe_scratch = './resources/imgs/img_severe_scratch_cropped.png',
                    ),
                    effect='crack',
                    thr=100,
                ),
                lid = dict(
                    effect='missed',
                    angle=[-10,53], # [deg]
                ),
                rad = dict(
                    effect=None,
                    angle=[]
                ),
            )

    def vis_objects(self, indices=np.arange(178, 188), conf_thr=0.0):
        dataset = build_dataset(self, split='test', cfg_effect=self.cfg_effect) # 'all', 'train', 'test'
        self.network.eval()

        with torch.no_grad():
            for idx_frame in tqdm(indices):
                print(f'* idx frame = {idx_frame}')
                dict_item = dataset[idx_frame]
                batch_dict = dataset.collate_fn([dict_item]) # single batch

                batch_dict['get_att_maps'] = None # att maps
                infer_mode = 'rlc' # ['rlc', 'rl', 'rc', 'lc', 'r', 'l', 'c']
                
                list_avail_feats = [] # order should be matched to KEY_FEATS in config (C -> L -> R)
                if 'c' in infer_mode:
                    list_avail_feats.append('cam_bev_feat')
                if 'l' in infer_mode:
                    list_avail_feats.append('spatial_features_2d')
                if 'r' in infer_mode:
                    list_avail_feats.append('bev_feat')

                # if 'r' in infer_mode:
                #     list_avail_feats.append('bev_feat')
                # if 'l' in infer_mode:
                #     list_avail_feats.append('spatial_features_2d')
                # if 'c' in infer_mode:
                #     list_avail_feats.append('cam_bev_feat')

                batch_dict['avail_feats'] = list_avail_feats

                output_dict = self.network(batch_dict)

                dict_item = dataset.get_label(dict_item)
                dict_item = dataset.get_ldr64(dict_item)
                dict_item = dataset.get_rdr_sparse(dict_item)

                pc_radar = dict_item['rdr_sparse']
                percentile_rate = 0.01 # [TBC] Change the percentile rate to the desired value.
                pc_radar = pc_radar[np.where(pc_radar[:,3]>np.quantile(pc_radar[:,3], 1-percentile_rate))[0],:]
                pred_dicts = output_dict['pred_dicts'][0]
                label = output_dict['label'][0]

                arr_x = np.arange(0, 72.0, 0.4) + 0.2
                arr_y = np.arange(-16.0, 16.0, 0.4) + 0.2

                self.show_bboxes_in_o3d(pc_lidar=dict_item['ldr64'], pc_radar=pc_radar, \
                                        pred_dicts=pred_dicts, label=label, confidence_thr=conf_thr)
                self.show_front_camera_img(dict_item, dataset.dict_t_params[dict_item['meta']['seq']]['front0']['radar2image'], \
                                           pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr)
                self.show_bboxes_in_plt(pc_lidar=dict_item['ldr64'], pc_radar=pc_radar, \
                                        pred_dicts=pred_dicts, label=label, confidence_thr=conf_thr)
                self.show_radar_tensor(arr_x, arr_y, dict_item,
                                       pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr)

                if 'get_att_maps' in output_dict.keys():
                    self.show_att_maps(arr_x, arr_y, output_dict['get_att_maps'],
                                         pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr, mode=infer_mode)
                plt.show()
                
                ### Vis each object feature map ###
                # obj_feat_vis = torch.sum(obj_feats, dim=1, keepdim=False).cpu().numpy()
                # for idx_obj in range(len(label)):
                #     plt.imshow(obj_feat_vis[idx_obj], cmap='viridis')
                # plt.show()
                ### Vis each object feature map ###

                if 'pointer' in output_dict.keys():
                    for dict_item in output_dict['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None

                for temp_key in output_dict.keys():
                    output_dict[temp_key] = None

                for temp_key in dict_item.keys():
                    dict_item[temp_key] = None

    def show_att_maps(self, arr_x, arr_y, att_maps,
                        pred_dicts=None, label=None, confidence_thr=0.0, title=None, mode='rlc'):
        print(att_maps.shape) # b, 3 (sensor: cam, lid, rad), n_repeat_channel, arr_y, arr_x
        # print(torch.sum(att_maps[0,:,:,:,:], axis=0)) # check if it is 1.
        plt.figure(figsize=(13, 4))
        """
        * cam as red, lid as green, rad as blue
        """
        # channel_idx = 0
        # img_in_channel = att_maps[0,:,channel_idx,:,:.permute(1, 2, 0).detach().cpu().numpy()] # 80, 180, 3
        img_mean_channel = torch.mean(att_maps[0,:,:,:,:], dim=1).permute(1, 2, 0).detach().cpu().numpy() # 80, 180, 3
        H, W, _ = img_mean_channel.shape

        list_cat_img = []
        # print(img_mean_channel.shape)
        start_idx = 0
        if 'c' in mode: # as red
            list_cat_img.append(img_mean_channel[:,:,start_idx:start_idx+1])
            start_idx += 1
        else:
            list_cat_img.append(np.zeros((H,W,1)))
        
        if 'l' in mode: # as green
            list_cat_img.append(img_mean_channel[:,:,start_idx:start_idx+1])
            start_idx += 1
        else:
            list_cat_img.append(np.zeros((H,W,1)))

        if 'r' in mode: # as blue
            list_cat_img.append(img_mean_channel[:,:,start_idx:start_idx+1])
            start_idx += 1
        else:
            list_cat_img.append(np.zeros((H,W,1)))

        img_cat = np.concatenate(list_cat_img, axis=2)

        plt.imshow(img_cat)

    def show_front_camera_img(self, dict_item, sensor2image,
                              pred_dicts=None, label=None, confidence_thr=0.0, title=None):
        dir_imgs = '/media/donghee/HDD_0/kradar_imgs/undistorted'
        cam_img = os.path.join(dir_imgs, dict_item['meta']['seq'], 'front0', dict_item['meta']['idx']['camf']+'.png')
        img_cam = plt.imread(cam_img)

        sensor2img = sensor2image.copy() # 4,4

        plt.imshow(img_cam)

        if pred_dicts is not None:
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            
            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                label_id = pred_labels[idx_pred]
                
                if score > confidence_thr:                    
                    # Draw box to front img
                    # Get 3D corners
                    corners_3d = get_3d_box_corners(x, y, z, l, w, h, th)

                    # Project corners to image
                    corners_img = project_points_to_image(corners_3d, sensor2img)

                    # Draw box on image
                    draw_3d_box_to_image(corners_img, color=self.colors_pred, linewidth=2)
        
        # Draw ground truth boxes
        if label is not None:
            for obj in label:
                cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
                
                # Draw box to front img
                # Project corners to image
                corners_img = project_points_to_image(corners_3d, sensor2img)

                # Draw box on image
                draw_3d_box_to_image(corners_img, color=self.colors_label, linewidth=2)

    def show_radar_tensor(self, arr_x, arr_y, dict_item,
                          pred_dicts=None, label=None, confidence_thr=0.0, title=None):
        plt.figure(figsize=(13, 4))
        path_cube = os.path.join(dict_item['meta']['header'], dict_item['meta']['seq'], 'radar_zyx_cube', 'cube_'+dict_item['meta']['idx']['rdr']+'.mat')
        arr_cube = np.flip(loadmat(path_cube)['arr_zyx'], axis=0)
        # arr_z_cb = np.arange(-30, 30, 0.4)+0.2
        # arr_y_cb = np.arange(-80, 80, 0.4)+0.2 # 160~240
        # arr_x_cb = np.arange(0, 100, 0.4)+0.2 # 0~180

        arr_cube = arr_cube[:,160:240,:180]
        cnt_non_zero = np.count_nonzero(arr_cube!=-1., axis=0) + 1
        arr_cube[arr_cube==-1] = 0.
        arr_cube = 10*np.log10(np.clip(np.sum(arr_cube, axis=0)/cnt_non_zero, 1., np.inf))
        arr_cube[arr_cube==0.] = -np.inf
        plt.pcolormesh(arr_x, arr_y, arr_cube, shading='auto', cmap='jet')
        plt.colorbar(label='Values')

        if pred_dicts is not None:
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            
            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                label_id = pred_labels[idx_pred]
                
                if score > confidence_thr:
                    # Create rotated rectangle
                    rect = get_rectangle_corners(x, y, l, w, th)
                    
                    # Draw box
                    plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor=self.colors_pred, linewidth=3, alpha=0.7)
                    
                    # Draw heading arrow
                    if self.show_details:
                        draw_heading_arrow(x, y, th, 'green', 2.0)
                    
                    # Display score and label
                    if self.show_details:
                        plt.text(x, y, f'{score:.2f}', 
                            color='green', fontsize=15, 
                            horizontalalignment='center',
                            verticalalignment='center')
        
        # Draw ground truth boxes
        if label is not None:
            for obj in label:
                cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
                
                # Create rotated rectangle
                rect = get_rectangle_corners(x, y, l, w, th)
                
                # Draw box
                plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor=self.colors_label, linewidth=3, alpha=0.7)
                
                # Draw heading arrow
                if self.show_details:
                    draw_heading_arrow(x, y, th, 'gray', 2.0)
                
                # Display class name and tracking ID
                if self.show_details:
                    plt.text(x, y, f'TrID:{trk_id}', 
                        color='gray', fontsize=15, 
                        horizontalalignment='center',
                        verticalalignment='center')
        
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')

        if title is not None:
            plt.title(title)
        else:
            plt.title('2D Heatmap')

        # Set plot range
        plt.xlim(0, 72)
        plt.ylim(-16, 16)

    def show_featuremap(self, arr_x, arr_y, bev_feat, \
                        pred_dicts=None, label=None, confidence_thr=0.0, title=None):
        
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(arr_x, arr_y, bev_feat, shading='auto', cmap='viridis')
        plt.colorbar(label='Values')

        # Draw predicted boxes
        if pred_dicts is not None:
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            
            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                label_id = pred_labels[idx_pred]
                
                if score > confidence_thr:
                    # Create rotated rectangle
                    rect = get_rectangle_corners(x, y, l, w, th)
                    
                    # Draw box
                    plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor='green', linewidth=2, alpha=0.8)
                    
                    # Draw heading arrow
                    draw_heading_arrow(x, y, th, 'green', 2.0)
                    
                    # Display score and label
                    plt.text(x, y, f'{score:.2f}', 
                            color='green', fontsize=15, 
                            horizontalalignment='center',
                            verticalalignment='center')
        
        # Draw ground truth boxes
        if label is not None:
            for obj in label:
                cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
                
                # Create rotated rectangle
                rect = get_rectangle_corners(x, y, l, w, th)
                
                # Draw box
                plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor='gray', linewidth=2, alpha=0.8)
                
                # Draw heading arrow
                draw_heading_arrow(x, y, th, 'gray', 2.0)
                
                # Display class name and tracking ID
                plt.text(x, y, f'TrID:{trk_id}', 
                        color='gray', fontsize=15, 
                        horizontalalignment='center',
                        verticalalignment='center')
        
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')

        if title is not None:
            plt.title(title)
        else:
            plt.title('2D Heatmap')

        # Set plot range
        plt.xlim(0, 72)
        plt.ylim(-16, 16)

    def show_bboxes_in_o3d(self, pc_lidar=None, pc_radar=None, pred_dicts=None, label=None, confidence_thr=0.0):
        """
        Show the bounding boxes in Open3D.

        Args:
            pc_lidar (np.ndarray): LiDAR point cloud. (N, 3+)
            pc_radar (np.ndarray): Radar point cloud. (N, 3+)
            pred_dicts (dict): Prediction dictionary.
            label (list): Label list.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if pc_lidar is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
            vis.add_geometry(pcd)

        if pc_radar is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_radar[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (len(pc_radar), 1)))
            vis.add_geometry(pcd)

        if pred_dicts is not None:
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                if score > confidence_thr:
                    draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[0.,1.,0.], radius=0.2)
                else:
                    continue

        for obj in label:
            cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
            draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[0.5,0.5,0.5], radius=0.05)
        
        vis.run()
        vis.destroy_window()

    def show_bboxes_in_plt(self, pc_lidar=None, pc_radar=None, pred_dicts=None, label=None, confidence_thr=0.0):
        """
        Show the bounding boxes in matplotlib BEV view.
        
        Args:
            pc_lidar (np.ndarray): LiDAR point cloud. (N, 3+)
            pc_radar (np.ndarray): Radar point cloud. (N, 3+)
            pred_dicts (dict): Prediction dictionary.
            label (list): Label list.
            confidence_thr (float): Confidence threshold for predictions.
        """
        plt.figure(figsize=(20, 10))
        
        # Plot point clouds
        if pc_lidar is not None:
            plt.scatter(pc_lidar[:, 0], pc_lidar[:, 1], 
                        c='gray', s=0.5, alpha=0.5, 
                        label='LiDAR Points')
        
        if pc_radar is not None:
            plt.scatter(pc_radar[:, 0], pc_radar[:, 1], 
                        c='red', s=2, alpha=0.7,
                        label='Radar Points')
        
        # Draw predicted boxes
        if pred_dicts is not None:
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            
            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                label_id = pred_labels[idx_pred]
                
                if score > confidence_thr:
                    # Create rotated rectangle
                    rect = get_rectangle_corners(x, y, l, w, th)
                    
                    # Draw box
                    plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor='green', linewidth=2, alpha=0.8)
                    
                    # Draw heading arrow
                    draw_heading_arrow(x, y, th, 'green', 2.0)
                    
                    # Display score and label
                    plt.text(x, y, f'{score:.2f}', 
                            color='green', fontsize=15, 
                            horizontalalignment='center',
                            verticalalignment='center')
        
        # Draw ground truth boxes
        if label is not None:
            for obj in label:
                cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
                
                # Create rotated rectangle
                rect = get_rectangle_corners(x, y, l, w, th)
                
                # Draw box
                plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor='gray', linewidth=2, alpha=0.8)
                
                # Draw heading arrow
                draw_heading_arrow(x, y, th, 'gray', 2.0)
                
                # Display class name and tracking ID
                plt.text(x, y, f'TrID:{trk_id}', 
                        color='gray', fontsize=15, 
                        horizontalalignment='center',
                        verticalalignment='center')
        
        # Set axes
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        # Set plot range
        plt.xlim(0, 72)
        plt.ylim(-16, 16)

        plt.legend()
        plt.title('Bird\'s Eye View')

    def validate_kitti_conditional(self):
        infer_mode = 'rlc' # ['rlc', 'rl', 'rc', 'lc', 'r', 'l', 'c']

        self.network.eval()
        list_conf_thr = [0.0, 0.3]

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

        str_local_time = get_local_time_str()
        str_exp = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)

        self.dataset_test = build_dataset(self, split='test') # 'all', 'train', 'test'

        class_names = []
        dict_label = self.dataset_test.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            if logit_idx > 0:
                class_names.append(k)
        self.dict_cls_id_to_name = dict()
        for idx_cls, cls_name in enumerate(class_names):
            self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        
        road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        time_cond_list = ['day', 'night']
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        # Check is_validate with small dataset
        is_shuffle = False
        tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')

        data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn, \
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        dir_epoch = 'none'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

            os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                f.write('')

            for road_cond in road_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                    f.write('')

            for time_cond in time_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                    f.write('')

            for weather_cond in weather_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                    f.write('')

            pred_dir_list = []
            label_dir_list = []
            desc_dir_list = []
            split_path_list = []

            ### For All Conditions ###
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dir_list.append(preds_dir)
            label_dir_list.append(labels_dir)
            desc_dir_list.append(desc_dir)
            split_path_list.append(split_path)
                            
            ### For Specific Conditions ###
            for road_cond in road_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)
                
                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for time_cond in time_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for weather_cond in weather_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

        # Creating gts and preds txt files for evaluation
        for idx_datum, dict_datum in enumerate(data_loader):
            try:                
                list_avail_feats = [] # order should be matched to KEY_FEATS in config (C -> L -> R)
                if 'c' in infer_mode:
                    list_avail_feats.append('cam_bev_feat')
                if 'l' in infer_mode:
                    list_avail_feats.append('spatial_features_2d')
                if 'r' in infer_mode:
                    list_avail_feats.append('bev_feat')

                dict_datum['avail_feats'] = list_avail_feats

                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            # if is_print_memory:
            #     print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                
            idx_name = str(idx_datum).zfill(6)
            
            road_cond_tag, time_cond_tag, weather_cond_tag = \
                dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
            # print(dict_out['desc'][0])

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                os.makedirs(labels_dir_road, exist_ok=True)
                os.makedirs(labels_dir_time, exist_ok=True)
                os.makedirs(labels_dir_weather, exist_ok=True)
                os.makedirs(desc_dir_road, exist_ok=True)
                os.makedirs(desc_dir_time, exist_ok=True)
                os.makedirs(desc_dir_weather, exist_ok=True)
                os.makedirs(preds_dir_road, exist_ok=True)
                os.makedirs(preds_dir_time, exist_ok=True)
                os.makedirs(preds_dir_weather, exist_ok=True)
                
                if is_feature_inferenced:    
                    pred_dicts = dict_out['pred_dicts'][0]
                    pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                    pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                    pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                    list_pp_bbox = []
                    list_pp_cls = []

                    for idx_pred in range(len(pred_labels)):
                        x, y, z, l, w, h, th = pred_boxes[idx_pred]
                        score = pred_scores[idx_pred]
                        
                        if score > conf_thr:
                            cls_idx = int(np.round(pred_labels[idx_pred]))
                            cls_name = class_names[cls_idx-1]
                            list_pp_bbox.append([score, x, y, z, l, w, h, th])
                            list_pp_cls.append(cls_idx)
                        else:
                            continue
                    pp_num_bbox = len(list_pp_cls)
                    dict_out_current = dict_out
                    dict_out_current.update({
                        'pp_bbox': list_pp_bbox,
                        'pp_cls': list_pp_cls,
                        'pp_num_bbox': pp_num_bbox,
                        'pp_desc': dict_out['meta'][0]['desc']
                    })
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                if dict_out_current is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                        if idx_label == 0:
                            mode = 'w'
                        else:
                            mode = 'a'

                        with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])

                    ### Process description ###
                    if len(dict_out_current['kitti_pred']) == 0:
                        with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                    else:
                        for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                            if idx_pred == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                    
                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
                    with open(split_path_road, 'a') as f:
                        f.write(str_log)
                    with open(split_path_time, 'a') as f:
                        f.write(str_log)
                    with open(split_path_weather, 'a') as f:
                        f.write(str_log)
                        
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
        for conf_thr in list_conf_thr:
            for condition in all_condition_list:
                try:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                    split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                    dt_annos = kitti.get_label_annos(preds_dir)
                    val_ids = read_imageset_file(split_path)
                    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                    list_metrics = []
                    list_results = []
                    for idx_cls_val in self.list_val_care_idx:
                        # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
                        # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
                        dict_metrics, result = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        list_metrics.append(dict_metrics)
                        list_results.append(result)
                    print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                    with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                        for dic_metric in list_metrics:
                            print('='*50)
                            print('Cls: ', dic_metric['cls'])
                            print('IoU:', dic_metric['iou'])
                            print('BEV: ', dic_metric['bev'])
                            print('3D: ', dic_metric['3d'])
                            print('-'*50)
                            
                            f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                            f.write('cls: ' + dic_metric['cls'] + '\n')
                            f.write('iou: ')
                            for iou in dic_metric['iou']:
                                f.write(str(iou) + ' ')
                            f.write('\n')
                            f.write('bev: ')
                            for bev in dic_metric['bev']:
                                f.write(str(bev) + ' ')
                            f.write('\n')
                            f.write('3d  :')
                            for det3d in dic_metric['3d']:
                                f.write(str(det3d) + ' ')
                            f.write('\n\n')
                    print('\n')
                except:
                    print('* Exception error (Pipeline): Samples for the codition are not found')

        path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
        print(f'* Check {path_check}')
        ### Validate per conf ###


if __name__ == '__main__':
    pipe_vis = ResultVis(path_cfg=PATH_CFG, path_pt=PATH_PT)
    # pipe_vis.vis_objects()
    pipe_vis.validate_kitti_conditional()
