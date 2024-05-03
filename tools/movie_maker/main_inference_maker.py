'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

# step1. inferencing all results
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import os.path as osp

### Hyper-params ###
PATH_CONFIG = './pretrained/RTNHPP/cfg_SFRTNH_2.yml'
PATH_MODEL = './pretrained/RTNHPP/model_10.pt'
### Hyper-params ###

try:
    from pipelines.pipeline_detection_v1_1 import PipelineDetection_v1_1
except:
    sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    from pipelines.pipeline_detection_v1_1 import PipelineDetection_v1_1

if __name__ == '__main__':
    pline = PipelineDetection_v1_1(PATH_CONFIG, mode='load_in_function')
    pline.infer_and_save_obj_results_to_directory(PATH_MODEL, is_train=False)
    pline.infer_and_save_obj_results_to_directory(PATH_MODEL, is_train=True)
