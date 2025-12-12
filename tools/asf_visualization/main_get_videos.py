import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'

import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os.path as osp

from PIL import Image
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
        self.cfg.DATASET.portion = None # [TBC] Change the portion to the desired seqs.
        ### [Custom Configurations] ###
        
        ### [Config for front cam] ###
        self.cfg.DATASET['t_params'] = dict()
        self.cfg.DATASET.t_params.load = True
        self.cfg.DATASET.t_params.dir = './resources/cam_calib/T_params_seq'
        self.cfg.DATASET.t_params.ref_sensor = 'radar'
        ### [Config for front cam] ###

        ### [Colors & Details] ###
        self.colors_pred  = [1., 1., 0.] # [0.1, 0.1, 0.1]
        self.colors_label = [0.6, 0., 0.]
        self.colors_camera = [0.8, 0.8, 0] # [0.5, 0.5, 0.5]
        self.show_details = False # arrow & conf score
        ### [Colors & Details] ###

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
                    effect=None, # None
                    thr=100,
                ),
                lid = dict(
                    effect='missed', # 'missed', # 'missed', # 'missed',
                    angle=[-22.5,22.5], # [deg]
                ),
                rad = dict(
                    effect=None,
                    angle=[]
                ),
            )
    
    def save_visualized_images(self, conf_thr=0.3, dir_save_folders=None):
        if dir_save_folders is None:
            print('* please set dir_save_folders')
            exit()
        else:
            list_folders_to_made = ['SAM', 'CAM', 'LID', 'RAD']
            for i in range(58):
                seq_folder = osp.join(dir_save_folders, f'{i+1}')
                os.makedirs(seq_folder, exist_ok=True)
                for folder_name in list_folders_to_made:
                    os.makedirs(osp.join(seq_folder, folder_name), exist_ok=True)
        
        dataset = build_dataset(self, split='all', cfg_effect=self.cfg_effect)
        self.network.eval()

        with torch.no_grad():
            for idx_frame in tqdm(range(len(dataset))):
                # print(f'* idx frame = {idx_frame}')
                dict_item = dataset[idx_frame]
                batch_dict = dataset.collate_fn([dict_item]) # single batch

                seq = dict_item['meta']['seq']
                frame = dict_item['meta']['idx']['rdr']

                batch_dict['get_att_maps'] = None # att maps
                infer_mode = 'rlc' # ['rlc', 'rl', 'rc', 'lc', 'r', 'l', 'c']
                
                list_avail_feats = [] # order should be matched to KEY_FEATS in config (C -> L -> R)
                if 'c' in infer_mode:
                    list_avail_feats.append('cam_bev_feat')
                if 'l' in infer_mode:
                    list_avail_feats.append('spatial_features_2d')
                if 'r' in infer_mode:
                    list_avail_feats.append('bev_feat')

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

                self.save_bboxes_in_plt(pc_lidar=dict_item['ldr64'], pc_radar=None, # pc_radar, \
                                        pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr, dir_save_folders=dir_save_folders, seq=seq, frame=frame)
                self.save_front_camera_img(dict_item, dataset.dict_t_params[dict_item['meta']['seq']]['front0']['radar2image'], \
                                           pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr, dir_save_folders=dir_save_folders, seq=seq, frame=frame)
                self.save_radar_tensor(arr_x, arr_y, dict_item,
                                       pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr, dir_save_folders=dir_save_folders, seq=seq, frame=frame)

                if 'get_att_maps' in output_dict.keys():
                    self.save_att_maps(arr_x, arr_y, output_dict['get_att_maps'],
                                         pred_dicts=pred_dicts, label=None, confidence_thr=conf_thr, mode=infer_mode, dir_save_folders=dir_save_folders, seq=seq, frame=frame)

                if 'pointer' in output_dict.keys():
                    for dict_item in output_dict['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None

                for temp_key in output_dict.keys():
                    output_dict[temp_key] = None

                for temp_key in dict_item.keys():
                    dict_item[temp_key] = None

                plt.close('all')

    def save_att_maps(self, arr_x, arr_y, att_maps,
                        pred_dicts=None, label=None, confidence_thr=0.0, title=None, mode='rlc', dir_save_folders=None, seq=None, frame=None):
        # print(att_maps.shape) # b, 3 (sensor: cam, lid, rad), n_repeat_channel, arr_y, arr_x
        # print(torch.sum(att_maps[0,:,:,:,:], axis=0)) # check if it is 1.
        plt.clf()
        plt.figure(figsize=(13, 4))
        """
        * cam as red, lid as green, rad as blue
        """
        # channel_idx = 0
        # img_in_channel = att_maps[0,:,channel_idx,:,:.permute(1, 2, 0).detach().cpu().numpy()] # 80, 180, 3
        img_mean_channel = torch.mean(att_maps[0,:,:,:,:], dim=1).permute(1, 2, 0).detach().cpu().numpy() # 80, 180, 3
        H, W, _ = img_mean_channel.shape

        ### [Calc availability] ###
        arg_indices = np.argmax(img_mean_channel, axis=2)
        total_bin = H*W
        ### [Calc availability] ###

        list_cat_img = []
        list_sum_vals = []
        # print(img_mean_channel.shape)
        start_idx = 0
        if 'c' in mode: # as red
            list_cat_img.append(img_mean_channel[:,:,start_idx:start_idx+1])
            # print('* camera for rank 1: ', (np.count_nonzero(arg_indices==start_idx)*100.)/(total_bin))
            # print('* camera > 0.3: ', (np.count_nonzero(img_mean_channel[:,:,start_idx:start_idx+1]>0.3)/(total_bin)))
            # print('* camera > 0.2: ', (np.count_nonzero(img_mean_channel[:,:,start_idx:start_idx+1]>0.2)/(total_bin)))
            sum_sensor = np.sum(img_mean_channel[:,:,start_idx:start_idx+1])
            # print('* camera sum: ', sum_sensor)
            list_sum_vals.append(sum_sensor)
            start_idx += 1
        else:
            list_cat_img.append(np.zeros((H,W,1)))
        
        if 'l' in mode: # as green
            list_cat_img.append(img_mean_channel[:,:,start_idx:start_idx+1])
            # print('* lidar for rank 1: ', (np.count_nonzero(arg_indices==start_idx)*100.)/(total_bin))
            # print('* lidar > 0.3: ', (np.count_nonzero(img_mean_channel[:,:,start_idx:start_idx+1]>0.3)*100./(total_bin)))
            # print('* lidar > 0.2: ', (np.count_nonzero(img_mean_channel[:,:,start_idx:start_idx+1]>0.2)*100./(total_bin)))
            sum_sensor = np.sum(img_mean_channel[:,:,start_idx:start_idx+1])
            # print('* lidar sum: ', sum_sensor)
            list_sum_vals.append(sum_sensor)
            start_idx += 1
        else:
            list_cat_img.append(np.zeros((H,W,1)))

        if 'r' in mode: # as blue
            list_cat_img.append(img_mean_channel[:,:,start_idx:start_idx+1])
            # print('* radar for rank 1: ', (np.count_nonzero(arg_indices==start_idx)*100.)/(total_bin))
            # print('* radar > 0.3: ', (np.count_nonzero(img_mean_channel[:,:,start_idx:start_idx+1]>0.3)*100./(total_bin)))
            # print('* radar > 0.2: ', (np.count_nonzero(img_mean_channel[:,:,start_idx:start_idx+1]>0.2)*100./(total_bin)))
            sum_sensor = np.sum(img_mean_channel[:,:,start_idx:start_idx+1])
            # print('* radar sum: ', sum_sensor)
            list_sum_vals.append(sum_sensor)
            start_idx += 1
        else:
            list_cat_img.append(np.zeros((H,W,1)))

        arr_sum_vals = np.array(list_sum_vals)
        # print('* utiliation rate: ', arr_sum_vals/np.sum(arr_sum_vals)*100.)

        img_cat = np.concatenate(list_cat_img, axis=2)
        img_cat = np.flipud(img_cat)

        if len(mode) == 1:
            img_cat = img_cat*0.6 # darker

        plt.imshow(img_cat)

        if pred_dicts is not None:
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            
            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                label_id = pred_labels[idx_pred]
                
                if score > confidence_thr:                    
                    rect = get_rectangle_corners(x, y, l, w, th)

                    rect_x = rect[:,0]
                    rect_y = rect[:,1]

                    rect_pixel_x = rect_x/72.*180.
                    rect_pixel_y = 80.-(rect_y+16.)/32.*80
                    
                    # Draw box
                    plt.fill(rect_pixel_x, rect_pixel_y, fill=False, edgecolor=self.colors_pred, linewidth=3, alpha=0.7)
        
        plt.xlim(0,180)
        plt.ylim(0,80)

        if dir_save_folders is not None:
            plt.savefig(osp.join(dir_save_folders, seq, 'SAM', f'{frame}.png'), dpi=300, format='png')
            plt.clf()

    def save_front_camera_img(self, dict_item, sensor2image,
                              pred_dicts=None, label=None, confidence_thr=0.0, title=None, dir_save_folders=None, seq=None, frame=None):
        plt.clf()
        dir_imgs = '/media/donghee/HDD_0/kradar_imgs/undistorted'
        cam_img = os.path.join(dir_imgs, dict_item['meta']['seq'], 'front0', dict_item['meta']['idx']['camf']+'.png')
        img_cam = plt.imread(cam_img)

        if self.cfg_effect is not None:
            cam_effect = self.cfg_effect['cam']['effect']
            if cam_effect is not None:
                thr_effect = self.cfg_effect['cam']['thr']
                img = Image.open(cam_img)
                img_effect = Image.open(self.cfg_effect['cam']['path_ori'][cam_effect])
                img_np = np.array(img)
                img_effect_np = np.array(img_effect)
                img_np[img_effect_np>thr_effect] = img_effect_np[img_effect_np>thr_effect]
                img_cam = Image.fromarray(img_np)
                img_np = None
                img_effect_np = None
                img_effect = None

        sensor2img = sensor2image.copy() # 4,4

        plt.imshow(img_cam, cmap='magma', vmin=0, vmax=1)

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
                    draw_3d_box_to_image(corners_img, color=self.colors_camera, linewidth=1.5)
        
        # Draw ground truth boxes
        if label is not None:
            for obj in label:
                cls_name, _, (x, y, z, th, l, w, h), trk_id = obj

                corners_3d = get_3d_box_corners(x, y, z, l, w, h, th)
                
                # Draw box to front img
                # Project corners to image
                corners_img = project_points_to_image(corners_3d, sensor2img)

                # Draw box on image
                draw_3d_box_to_image(corners_img, color=self.colors_label, linewidth=1.5)

        if dir_save_folders is not None:
            plt.xlim((0,1280))
            plt.ylim((720,0))
            plt.savefig(osp.join(dir_save_folders, seq, 'CAM', f'{frame}.png'), dpi=300, format='png')
            plt.clf()

    def save_radar_tensor(self, arr_x, arr_y, dict_item,
                          pred_dicts=None, label=None, confidence_thr=0.0, title=None, dir_save_folders=None, seq=None, frame=None):
        plt.clf()
        plt.figure(figsize=(13, 4))
        path_cube = os.path.join(dict_item['meta']['header'], dict_item['meta']['seq'], 'radar_zyx_cube', 'cube_'+dict_item['meta']['idx']['rdr']+'.mat')
        arr_cube = np.flip(loadmat(path_cube)['arr_zyx'], axis=0)

        arr_cube = arr_cube[:,160:240,:180]
        cnt_non_zero = np.count_nonzero(arr_cube!=-1., axis=0) + 1
        arr_cube[arr_cube==-1] = 0.
        arr_cube = 10*np.log10(np.clip(np.sum(arr_cube, axis=0)/cnt_non_zero, 1., np.inf))
        arr_cube[arr_cube==0.] = -np.inf
        plt.pcolormesh(arr_x, arr_y, arr_cube, shading='auto', cmap='viridis')
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
                    plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor=self.colors_pred, linewidth=5, alpha=0.9)
                    
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
                plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor=self.colors_label, linewidth=5, alpha=0.9)
                
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

        if dir_save_folders is not None:
            plt.savefig(osp.join(dir_save_folders, seq, 'RAD', f'{frame}.png'), dpi=300, format='png')
            plt.clf()

    def save_bboxes_in_plt(self, pc_lidar=None, pc_radar=None, pred_dicts=None, label=None, confidence_thr=0.0, show_details=False, dir_save_folders=None, seq=None, frame=None):
        """
        Show the bounding boxes in matplotlib BEV view.
        
        Args:
            pc_lidar (np.ndarray): LiDAR point cloud. (N, 3+)
            pc_radar (np.ndarray): Radar point cloud. (N, 3+)
            pred_dicts (dict): Prediction dictionary.
            label (list): Label list.
            confidence_thr (float): Confidence threshold for predictions.
        """
        plt.clf()
        plt.figure(figsize=(20, 10))
        
        # Plot point clouds
        if pc_lidar is not None:
            if pc_radar is not None:
                plt.scatter(pc_lidar[:, 0], pc_lidar[:, 1], 
                            c='gray', s=0.5, alpha=0.5, 
                            label='LiDAR Points')
            elif pc_radar is None:
                # color_map = (pc_lidar[:,2] - np.min(pc_lidar[:,2]))/(np.max(pc_lidar[:,2])-np.min(pc_lidar[:,2]))

                color_map = pc_lidar[:,2].copy()
                color_map[np.where(color_map>np.median(color_map))] = np.median(color_map)
                plt.gca().set_facecolor((0.95,0.95,0.95))
                plt.scatter(pc_lidar[:, 0], pc_lidar[:, 1], 
                            c=color_map, cmap='Blues', s=5.0, alpha=0.7)
        
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
                    plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor=self.colors_pred, linewidth=10, alpha=0.9)
                    
                    if show_details:
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
                plt.fill(rect[:, 0], rect[:, 1], fill=False, edgecolor=self.colors_label, linewidth=10, alpha=0.9)
                
                if show_details:
                    # Draw heading arrow
                    draw_heading_arrow(x, y, th, 'gray', 2.0)
                    
                    # Display class name and tracking ID
                    plt.text(x, y, f'TrID:{trk_id}', 
                            color=self.colors_label, fontsize=15, 
                            horizontalalignment='center',
                            verticalalignment='center')
        
        # Set axes
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        # Set plot range
        plt.xlim(0, 72)
        plt.ylim(-16, 16)

        # plt.legend()
        plt.title('Bird\'s Eye View')

        if dir_save_folders is not None:
            plt.savefig(osp.join(dir_save_folders, seq, 'LID', f'{frame}.png'), dpi=300, format='png')
            plt.clf()

if __name__ == '__main__':
    pipe_vis = ResultVis(path_cfg=PATH_CFG, path_pt=PATH_PT)
    pipe_vis.save_visualized_images(dir_save_folders='/media/donghee/HDD_0/K-Radar_fusion_DH/png_ASF_vis')
