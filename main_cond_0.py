'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

### Change here ###
EXP_NAME = 'exp_230115_000000_RTNH'
MODEL_EPOCH = 10 
# SAMPLE_INDICES = [0, 2, 4, 6, 10]
# CONFIDENCE_THR = 0.3
### Change here ###

if __name__ == '__main__':
    PATH_CONFIG = f'./logs/{EXP_NAME}/config.yml'
    PATH_MODEL = f'./logs/{EXP_NAME}/models/model_{MODEL_EPOCH}.pt'

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='test')

    ### Save this file for checking ###
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    ### Save this file for checking ###

    pline.load_dict_model(PATH_MODEL)
    # pline.vis_infer(sample_indices=SAMPLE_INDICES, conf_thr=CONFIDENCE_THR)
    # pline.validate_kitti(list_conf_thr=[0.3, 0.5], is_subset=False)
    pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
    
