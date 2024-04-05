<p align="center">
  <img src = "./docs/imgs/readme_logo.png" width="60%">
</p>

`KAIST-Radar (K-Radar)` (provided by ['AVELab'](http://ave.kaist.ac.kr/)) is a novel large-scale object detection dataset and benchmark that contains 35K frames of 4D Radar tensor (4DRT) data with power measurements along the Doppler, range, azimuth, and elevation dimensions, together with carefully annotated 3D bounding box labels of objects on the roads. K-Radar includes challenging driving conditions such as adverse weathers (fog, rain, and snow) on various road structures (urban, suburban roads, alleyways, and highways). In addition to the 4DRT, we provide auxiliary measurements from carefully calibrated high-resolution Lidars, surround stereo cameras, and RTK-GPS. This repository provides the `K-Radar` dataset, annotation tool for 3d bounding boxes, and the visualization tool for showing the inference results and calibrating the sensors.

![image](./docs/imgs/kradar_examples.png)

The URLs listed below are useful for using the K-Radar dataset and benchmark:
* <a href="https://arxiv.org/abs/2206.08171"> K-Radar paper and appendix [NeurIPS22] </a>
* <a href="https://ieeexplore.ieee.org/document/10186820"> Enhanced K-Radar [IV23] </a>
* <a href=""> Repository of Auto-labeling for 4D Radar </a>
* <a href=""> Repository of Odometry for 4D Radar </a>
* <a href="https://www.youtube.com/watch?v=TZh5i2eLp1k"> The video clip that shows each sensor measurement dynamically changing during driving under the heavy snow condition </a>
* <a href="https://www.youtube.com/watch?v=ylG0USHCBpU"> The video clip that shows the 4D radar tensor & Lidar point cloud (LPC) calibration and annotation process </a>
* <a href="https://www.youtube.com/watch?v=ILlBJJpm4_4"> The video clip that shows the annotation process in the absence of LPC measurements of objects  </a>
* <a href="https://www.youtube.com/watch?v=U4qkaMSJOds"> The video clip that shows calibration results </a>
* <a href="https://www.youtube.com/watch?v=MrFPvO1ZjTY"> The video clip that shows the GUI-based program for visualization and neural network inference </a>
* <a href="https://www.youtube.com/watch?v=8mqxf58_ZAk"> The video clip that shows the information on tracking for multiple objects on the roads </a>

## License and Commercialization Inquiries
The `K-Radar` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.

All technologies in this repository have been developed by [`AVELab`](http://ave.kaist.ac.kr/) and are being commercialized by `Zeta Mobility`. For commercialization inquiries, please contact `Zeta Mobility` (URL and e-mail will be provided).

# K-Radar Dataset
For the preparation of the dataset and pipeline, please refer to the following document: [Dataset Preparation Guide](/docs/dataset.md)

We provide the K-Radar dataset in three ways to ensure effective deployment of the large-sized data (total 16TB):

1. Access to the total dataset via our local server
2. A portion of the dataset via Google Drive
3. The total dataset via shipped HDD (Hard Disk Drive)

For more details, please refer to [the dataset documentation](/docs/dataset.md).

## Detection

This is the documentation for how to use our detection frameworks with K-Radar dataset. We tested the K-Radar detection frameworks on the following environment:

* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32
* open3d 0.15.2

### Requirements

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

### Train & Evaluation
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

### Model Zoo
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

## Pre-processing

<a href="https://drive.google.com/file/d/1hTHAHEYdhl6yEDdTz3uxv08vsdG4lxu7/view?usp=drive_link">Link</a>
Wide range, Quantile (RTNH) / 2 stage CFAR for Sidelobe filtering (RTNH+): code will be uploaded soon. (gif)
Function: datasetv1_1 generate~
Density: ref Enhanced
TODO

## Auto-labeling

TODO

## Odometry
We provide the location of a GPS antenna, essential for accurate ground-truth odometry. This location is precisely processed by integrating data from high-resolution LiDAR, RTK-GPS, and IMU data. To ensure the utmost accuracy, we verify the vehicle's location by correlating the LiDAR sensor data against a detailed, high-resolution map, as illustrated below. For security purposes, we present this location information in local coordinates rather than global coordinates (i.e., UTM). The data of Sequence 1 is accessible in the `resources/odometry` directory. The repository for the odometry will be made available within the next few weeks.

![image](./docs/imgs/processed_odometry.png)

## Acknowledgement
The K-Radar dataset is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), [Sangjae Cho](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=3), [Hong-Woo Seok](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=25), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

We thank the maintainers of the following projects that enable us to develop `K-Radar`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`Rotated_IoU`](https://github.com/lilanxiao/Rotated_IoU) by lilanxiao, and [`kitti-object-eval-python`](https://github.com/traveller59/kitti-object-eval-python) by traveller59.

We extend our gratitude to Jen-Hao Cheng, Sheng-Yao Kuan, Aishi Huang, Hou-I Liu, Christine Wu and Wenzheng Zhao in [the Information Processing Lab at the University of Washington](https://ipl-uw.github.io/) for providing [the refined K-Radar label](https://drive.google.com/file/d/1hEN7vdqoLaHZFf7woj5nj4hjpP88f7gU/view?usp=sharing).

This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 01210790) and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2021R1A2C3008370).

## Citation
If you find this work is useful for your research, please consider citing:
```
@inproceedings{
  paek2022kradar,
  title={K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions},
  author={Dong-Hee Paek and Seung-Hyun Kong and Kevin Tirta Wijaya},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=W_bsDmzwaZ7}
}
```
