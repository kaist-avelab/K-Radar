# Detection

This is the documentation for how to use our detection frameworks with K-Radar dataset. We tested the K-Radar detection frameworks on the following environment:

* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32
* open3d 0.15.2

## Requirements

1. Clone the repository
```
git clone https://github.com/kaist-avelab/K-Radar.git
cd K-Radar
```

2. Create a conda environment
```
conda create -n kradar python=3.8.13 -y
conda activate kradar
```

3. Install PyTorch (We recommend pytorch 1.11.0.)

4. Install the dependencies
```
pip install -r requirements.txt
```

5. Build packages for Rotated IoU
```
cd utils/Rotated_IoU/cuda_op
python setup.py install
```

6. Modify the code in packages
```
Add line 11: 'from .nms import rboxes' for __init__.py of nms module.
Add line 39: 'rrect = tuple(rrect)' and comment line 41: 'print(r)' in nms.py of nms module.
```

7. Build packages for OpenPCDet operations
```
cd ../../../ops
python setup.py develop
```

8. Unzip 'kradar_revised_label_v2_0.zip' in the 'tools/revise_label' directory (For the updated labeling format, please refer to [the dataset documentation](/docs/dataset.md).)

We use the operations from <a href="https://github.com/open-mmlab/OpenPCDet">OpenPCDet</a> repository and acknowledge that all code in `ops` directory is sourced from there.
To align with our project requirements, we have made several modifications to the original code and have uploaded the revised versions to our repository.
We extend our gratitude to MMLab for their great work.

## Train & Evaluation
* To train the model, prepare the total dataset and run
```
python main_train_0.py
```

* To evaluate the model, modify the path and run
```
python main_val_0.py (for evaluation)
python main_cond_0.py (for conditional evaluation)
```

* To visualize the inference result, modify the path and run
```
python main_test_0.py (with code)
python main_vis.py (with GUI)
```

## Model Zoo
The reported values are ${AP_{3D}}$ for Sedan class. (based on the label v1.0)
|Name|Total|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|Pretrained|Logs|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|RTNH|47.4|49.9|56.7|52.8|42.0|41.5|50.6|44.5|<a href="https://drive.google.com/drive/folders/1Z9f0yiddLKkygkg-QKBsrCDK2OvNWGIA?usp=drive_link">Link</a>|<a href="https://drive.google.com/file/d/1b4DTwO_SnKOfPrn1F6UQZ-dy4Gi91Eun/view?usp=drive_link">Link</a>|
|RTN|40.1|45.6|48.8|46.9|32.9|22.6|36.8|36.8|<a href="https://drive.google.com/drive/folders/1Z9f0yiddLKkygkg-QKBsrCDK2OvNWGIA?usp=drive_link">Link</a>|<a href="https://drive.google.com/file/d/1b4DTwO_SnKOfPrn1F6UQZ-dy4Gi91Eun/view?usp=drive_link">Link</a>|
|PILLARS|45.4|52.3|61.0|42.2|44.5|22.7|40.6|29.7|<a href="https://drive.google.com/drive/folders/1Z9f0yiddLKkygkg-QKBsrCDK2OvNWGIA?usp=drive_link">Link</a>|<a href="https://drive.google.com/file/d/1b4DTwO_SnKOfPrn1F6UQZ-dy4Gi91Eun/view?usp=drive_link">Link</a>|

Based on our findings presented, we elucidate **the capability of the 4D Radar in terms of 3D object detection and its resilience in the face of adverse weather conditions**. This is evident from its relatively stable performance under conditions of sleet and heavy snow, especially when compared to the performance metrics of LiDAR neural networks. We **do not claim** that 4D Radar is always superior to LiDAR in 3D object detection. In other words, we note that the performance of LiDAR neural networks could surpass that of the 4D Radar neural network with improvements in its structure or hyper-parameters.

### Performance of RTNH on wider areas and multiple classes (Sedan & Bus or Truck)
The performance metrics presented below have three main discrepancies compared to the results reported earlier (found in Appendix H and Section 4 of the K-Radar paper, respectively):

1. The evaluation now considers a broader area (with x, y, z each ranging from 0 to 72, -16 to 16, and -2 to 7.6 meters), unlike the narrower range (x, y, z each from 0 to 72, -6.4 to 6.4, and -2 to 6 meters) used previously.
2. The classes have been expanded to include not only the `Sedan` class but also the `Bus or Truck` class, representing compact and large vehicles, respectively.
3. The performance analysis has been updated to incorporate the version 2.0 labels. For changes relative to the version 1.0 labels, please refer to the `Revising K-Radar label` section.

For a more detailed performance review (e.g., performance for each road environment), please refer to the logs at <a href="https://drive.google.com/drive/folders/11x45ozlnaLDe6plEACi55UwFVSAcwlNW?usp=drive_link">this link</a>: available at log_test_RTNH_wide_11epoch.zip.

For the model below, please check the configuration in configs/cfg_RTNH_wide.yml and <a herf="https://drive.google.com/file/d/1ZMtq9BiWCHKKOc20pClCHyhOEI2LLiNM/view?usp=drive_link">the pretrained model</a> in the Google Drive. (We have updated the RTNH head code within the same anchor head format provided by <a href="https://github.com/open-mmlab/OpenPCDet">OpenPCDet</a> to enhance its stability and compatibility. In addition, we denote the performance of the `Bus or Truck` class with '-' in both Rain and Sleet conditions, as they do not exist in these conditions. 

(1) ${AP_{3D}}$
| Class        | Total | Normal | Overcast | Fog  | Rain | Sleet | LightSnow | HeavySnow |
|:------------:|:-----:|:------:|:--------:|:----:|:----:|:-----:|:---------:|:---------:|
| Sedan        | 48.2  | 45.5   | 58.8     | 79.3 | 40.3 | 48.1  | 65.6      | 52.6      |
| Bus or Truck | 34.4  | 25.3   | 31.1     | -    | -    | 28.5  | 78.2      | 46.3      |

(2) ${AP_{BEV}}$
| Class        | Total | Normal | Overcast | Fog  | Rain | Sleet | LightSnow | HeavySnow |
|:------------:|:-----:|:------:|:--------:|:----:|:----:|:-----:|:---------:|:---------:|
| Sedan        | 56.7  | 53.8   | 68.3     | 89.6 | 49.3 | 55.6  | 69.4      | 60.3      |
| Bus or Truck | 45.3  | 31.8   | 32.0     | -    | -    | 34.4  | 89.3      | 78.0      |
