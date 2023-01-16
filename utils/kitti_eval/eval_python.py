import kitti_common as kitti
from eval import get_official_eval_result
import numpy as np

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

PATH_HEADER = "/home/oem/Donghee/K-Radar/logs/exp_230109_190313_RTNH/test_kitti/none/0.5"

if __name__ == '__main__':
    path_header = PATH_HEADER
    det_path = path_header+"/pred"
    dt_annos = kitti.get_label_annos(det_path)
    gt_path = path_header+"/gt"
    gt_split_file = path_header+"/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
    val_image_ids = _read_imageset_file(gt_split_file)
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    dict_metrics, eval_sed = get_official_eval_result(gt_annos, dt_annos, 0, iou_mode='easy', is_return_with_dict=True)
    print(eval_sed)
    print(dict_metrics)
