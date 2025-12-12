import cv2
import numpy as np
import os.path as osp
import os
import yaml
import copy
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

def get_matrices_from_dict_calib(dict_calib):
    img_size = (dict_calib['img_size_w'], dict_calib['img_size_h'])
    intrinsics = np.array([
        [dict_calib['fx'], 0.0, dict_calib['px']],
        [0.0, dict_calib['fy'], dict_calib['py']],
        [0.0, 0.0, 1.0]
    ])
    distortion = np.array([
        dict_calib['k1'], dict_calib['k2'], dict_calib['k3'], \
        dict_calib['k4'], dict_calib['k5']
    ]).reshape((-1,1))

    # L to C
    yaw_ldr2cam = dict_calib['yaw_ldr2cam']
    pitch_ldr2cam = dict_calib['pitch_ldr2cam']
    roll_ldr2cam = dict_calib['roll_ldr2cam']
    r_ldr2cam = (R.from_euler('zyx', [yaw_ldr2cam, pitch_ldr2cam, roll_ldr2cam], degrees=True)).as_matrix()

    x_ldr2cam = dict_calib['x_ldr2cam']
    y_ldr2cam = dict_calib['y_ldr2cam']
    z_ldr2cam = dict_calib['z_ldr2cam']
    T_ldr2cam = np.concatenate([r_ldr2cam, np.array([x_ldr2cam,y_ldr2cam,z_ldr2cam]).reshape(-1,1)], axis=1)
    # L to C
    
    return img_size, intrinsics, distortion, T_ldr2cam

dict_args = dict(
    dir_save = '/media/donghee/HDD_2/kradar_imgs_all/cropped',
    cam = 'front0',
    dir_dataset = '/media/donghee/kradar/dataset',
    ori_shape = (1280,720),
    resize = 0.7,
    crop = (96,170,800,426),
    cam_calib = './resources/cam_calib/calib_seq',
)

if __name__ == '__main__':
    cfg = EasyDict(dict_args)
    dir_save = cfg.dir_save
    dir_dataset = cfg.dir_dataset

    key_cam = cfg.cam
    cam_header = 'cam-' + cfg.cam[:-1]
    stereo_idx = int(cfg.cam[-1])

    ori_shape = cfg.ori_shape
    resize = cfg.resize
    crop = cfg.crop

    W, H = ori_shape
    resize_dims = (int(W * resize), int(H * resize))

    ### Load dict cam calib ###
    def get_dict_cam_calib_from_yml_all_seqs():
        dict_cam_calib_seqs = dict()
        for idx in range(58):
            dict_cam_calib_seqs.update({f'{idx+1}': None})
        
        list_seqs = sorted(os.listdir(cfg.cam_calib))
        list_seqs_int = list(map(lambda x: int(x.split('_')[1]), list_seqs))
        
        list_cams = ['front0', 'front1', 'right0', 'right1', 'rear0', 'rear1', 'left0', 'left1']
        #            'cam_1.yml', 'cam_2.yml', ... , 'cam_8.yml'

        for seq_int in list_seqs_int:
            seq_str = f'{seq_int}'
            dict_cam_calib_seqs[seq_str] = dict()
            dir_seq = osp.join(cfg.cam_calib, 'seq_' + seq_str.zfill(2))
            list_ymls = sorted(os.listdir(dir_seq))

            assert len(list_ymls) == 8, '* check # of ymls'

            for idx_yml, yml_file in enumerate(list_ymls):
                key_cam = list_cams[idx_yml]
                path_yml = osp.join(dir_seq, yml_file)
                with open(osp.join(path_yml), 'r') as yml_opened:
                    dict_temp = yaml.safe_load(yml_opened)

                if 'img_size_w' not in dict_temp.keys():
                    dict_temp['img_size_w'] = 1280
                    dict_temp['img_size_h'] = 720

                dict_cam_calib_seqs[seq_str].update({key_cam: get_matrices_from_dict_calib(dict_temp)})
                # img_size, intrinsics, distortion, T_ldr2cam
        
        return dict_cam_calib_seqs

    dict_cam_calib = get_dict_cam_calib_from_yml_all_seqs()

    for idx_seq in range(58):
        seq = f'{idx_seq+1}'
        dir_save_seq = osp.join(dir_save, seq, key_cam)
        os.makedirs(dir_save_seq, exist_ok=True)

        dir_img = osp.join(dir_dataset, seq, cam_header)
        list_imgs = sorted(os.listdir(dir_img))

        img_size_ret, intrinsics_ret, distortion_ret, T_ldr2cam_ret = dict_cam_calib[seq][key_cam]

        print(f'* Processing {seq}/58...')
        for img_file in tqdm(list_imgs):
            img_loaded = cv2.imread(osp.join(dir_img, img_file))
            if stereo_idx == 0:
                img = img_loaded[:,:1280,:]
            else:
                img = img_loaded[:,1280:,:]

            img_size = copy.deepcopy(img_size_ret)
            intrinsics = copy.deepcopy(intrinsics_ret)
            distortion = copy.deepcopy(distortion_ret)
            T_ldr2cam = copy.deepcopy(T_ldr2cam_ret)

            # undistort
            ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
                
            for j in range(3):
                for i in range(3):
                    intrinsics[j,i] = ncm[j, i]
            
            map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1)
            img_temp = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

            img_undistorted = Image.fromarray(np.flip(img_temp, axis=2)) # bgr to rgb
            img_process = img_undistorted.resize(resize_dims)
            img_process = img_process.crop(crop)

            cam_idx = img_file.split('_')[1].split('.')[0]
            img_process.save(osp.join(dir_save_seq, f'{cam_idx}.png'))
