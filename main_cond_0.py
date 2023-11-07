'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_MODEL = './pretrained/RTNH_wide_10.pt'

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='test')
    pline.load_dict_model(PATH_MODEL)
    
    pline.network.eval()

    # Save the code for identification
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    
    pline.validate_kitti_conditional(list_conf_thr=[0.3, 0.5, 0.7], is_subset=False, is_print_memory=False)
