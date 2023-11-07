'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SAMPLE_INDICES = [10,11,12,30,70,95,150]
CONFIDENCE_THR = 0.5

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_MODEL = './pretrained/RTNH_wide_10.pt'

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='vis')
    pline.load_dict_model(PATH_MODEL)
    
    pline.network.eval()
    
    import torch
    import open3d as o3d
    from tqdm import tqdm
    from torch.utils.data import Subset

    dataset_loaded = pline.dataset_test
    subset = Subset(dataset_loaded, SAMPLE_INDICES)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = pline.dataset_test.collate_fn,
            num_workers = pline.cfg.OPTIMIZER.NUM_WORKERS)
    
    for dict_item in tqdm(data_loader):
        dict_item = pline.network(dict_item)

        dataset = pline.dataset_test
        pc_lidar = dataset.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])

        class_names = []
        dict_label = dataset.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            if logit_idx > 0:
                class_names.append(k)

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
        vis.add_geometry(pcd)

        label = dict_item['label'][0]
        for obj in label:
            cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
            _, _, rgb, _ = dataset.label['Label']
            dataset.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
        
        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        for idx_pred in range(len(pred_labels)):
            x, y, z, l, w, h, th = pred_boxes[idx_pred]
            score = pred_scores[idx_pred]
            
            if score > CONFIDENCE_THR:
                cls_idx = pred_labels[idx_pred]
                cls_name = class_names[cls_idx-1]
                _, _, rgb, _ = dataset.label[cls_name]
                dataset.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
            else:
                continue

        vis.run()
        vis.destroy_window()
