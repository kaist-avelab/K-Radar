<p align="center">
  <img src = "./docs/imgs/readme_logo.png" width="60%">
</p>

`KAIST-Radar (K-Radar)` (provided by ['AVELab'](http://ave.kaist.ac.kr/)) is a novel large-scale object detection dataset and benchmark that contains 35K frames of 4D Radar tensor (4DRT) data with power measurements along the Doppler, range, azimuth, and elevation dimensions, together with carefully annotated 3D bounding box labels of objects on the roads. K-Radar includes challenging driving conditions such as adverse weathers (fog, rain, and snow) on various road structures (urban, suburban roads, alleyways, and highways). In addition to the 4DRT, we provide auxiliary measurements from carefully calibrated high-resolution Lidars, surround stereo cameras, and RTK-GPS. This repository provides the `K-Radar` dataset, annotation tool for 3d bounding boxes, and the visualization tool for showing the inference results and calibrating the sensors.

![image](./docs/imgs/kradar_examples.png)

The URLs listed below are useful for using the K-Radar dataset and benchmark:
* <a href="https://arxiv.org/abs/2206.08171"> K-Radar paper and appendix [arxiv] </a>
* <a href="https://www.youtube.com/watch?v=TZh5i2eLp1k"> The video clip that shows each sensor measurement dynamically changing during driving under the heavy snow condition </a>
* <a href="https://www.youtube.com/watch?v=ylG0USHCBpU"> The video clip that shows the 4D radar tensor & Lidar point cloud (LPC) calibration and annotation process </a>
* <a href="https://www.youtube.com/watch?v=ILlBJJpm4_4"> The video clip that shows the annotation process in the absence of LPC measurements of objects  </a>
* <a href="https://www.youtube.com/watch?v=U4qkaMSJOds"> The video clip that shows calibration results </a>
* <a href="https://www.youtube.com/watch?v=MrFPvO1ZjTY"> The video clip that shows the GUI-based program for visualization and neural network inference </a>
* <a href="https://www.youtube.com/watch?v=8mqxf58_ZAk"> The video clip that shows the information on tracking for multiple objects on the roads </a>

# K-Radar Dataset
This is the documentation for how to use our detection frameworks with K-Radar dataset.
We tested the K-Radar detection frameworks on the following environment:
* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32
* open3d 0.15.2

## Notice
[2022-10-14] We have realized that the network-attached storage (NAS) download link is unstable. As such, we are preparing the Google Drive download link similar to our other project <a href="https://github.com/kaist-avelab/K-Lane">K-Lane</a>.

[2022-09-30] The `K-Radar` dataset is made available via a network-attached storage (NAS) <a href="http://QuickConnect.to/kaistavelab">download link</a>.

[2022-12-01] The `K-Radar` dataset is being uploaded to Google Drive. It will take time to upload our entire dataset.

[2022-12-06] We are checking and fixing the NAS server after realizing it has been down.

[2022-12-16] We have an unidentified problem in allocating IP to our NAS server. We apologize for this inconvenience.

[2022-12-19] The NAS server's URL has changed and is now accessible.

[2023-01-16] The pipepline for `K-Radar` dataset is updated, and the previous repository has been depracated.

## Preparing the Dataset

* Via our server

1. To download the dataset, log in to <a href="http://QuickConnect.to/kaistavelab"> our server </a> with the following credentials: 
      ID       : kradards
      Password : Kradar2022
2. Go to the "File Station" folder, and download the dataset by right-click --> download.
   Note for Ubuntu user, there might be some error when unzipping the files. Please check the "readme_to_unzip_file_in_linux_system.txt".
3. After all files are downloaded, please arrange the workspace directory with the following structure:
```
KRadarFrameworks
      ├── tools
      ├── configs
      ├── datasets
      ├── models
      ├── pipelines
      ├── resources
      ├── uis
      ├── utils
      ├── logs
├── kradar_dataset (We handle the K-Radar dataset with multiple HDDs due to its large capacity.)
      ├── dir_2to5
            ├── 2
            ...
            ├── 5
      ├── dir_1to20
            ├── 1
            ├── 6
            ...
            ├── 20
      ├── dir_21to37
            ├── 21
            ...
            ├── 37
      ├── dir_38to58
            ├── 38
            ...
            ├── 58
```

* Via Google Drive Urls

The `K-Radar` dataset is being uploaded to Google Drive. It will take time to upload our entire dataset.
The 4D Radar tensor is only available on our server because it is too large to upload to Google Drive. (We can charge for up to 2TB of Google Drive storage space.)
We provide camera images, Lidar point cloud, RTK-GPS, and Radar tensor (for network input) via Google Drive for flexibility.

## Sequence composition

Each of the 58 sequences consists of listed files or folders as follows.
We add the explanation or usage next to each file or folder.
```
Sequence_number.zip (e.g. 1.zip)
├── cam-front: Front camera images
├── cam-left: Left camera images
├── cam-rear: Rear camera images
├── cam-right: Right camera images
├── cell_path: File to generate data from bag or radar ADC files (refer to the matlab file generation code.)
├── description.txt: Required for conditional evaluation (e.g., weather conditions)
├── info_calib: Calibration info btwn 4D radar and Lidar
├── info_label: Label of frames in txt
├── info_label_rev: not used
├── lidar_bev_image: Used for the labeling program
├── lidar_bev_image_origin: not used
├── lidar_point_cloud: Used for the labeling program
├── os1-128: High resolution mid-range Lidar point cloud
├── os2-64: Long-range Lidar point cloud (mainly used)
├── radar_bev_image: Used for the labeling program
├── radar_bev_image_origin: not used
├── radar_tesseract: 4D (DRAE) Radar tensor (250MB per frame) 
├── radar_zyx_cube: 3D (ZYX) Radar tensor (used for Sparse tensor generation)
├── time_info: Used for calibration
```

## Requirements

1. Clone the repository
```
git clone https://github.com/kaist-avelab/K-Radar.git
cd K-Radar
```

2. Create a conda environment
```
conda create -n kradar python=3.8.13
conda activate kradar
```

3. Install PyTorch (We recommend pytorch 1.10.0.)

4. Install the dependencies
```
pip install -r requirements.txt
```

5. Build packages for kradar
```
cd utils/Rotated_IoU/cuda_op
ptyhon setup.py install
```

6. Modify the code in packages
```
Add line 11: 'from .nms import rboxes' for __init__.py of nms module.
Add line 39: 'rrect = tuple(rrect)' and comment line 41: 'print(r)' in nms.py of nms module.
```

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
The reported values are ${AP_{3D}}$ for `Sedan` class.
|Name|Total|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|Pretrained|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|RTNH|47.4|49.9|56.7|52.8|42.0|41.5|50.6|44.5|<a href="TBD1">Link</a>|
|RTN|40.1|45.6|48.8|46.9|32.9|22.6|36.8|36.8|<a href="TBD2">Link</a>|
|PILLARS|45.4|52.3|61.0|42.2|44.5|22.7|40.6|29.7|<a href="TBD3">Link</a>|

## License
The `K-Radar` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.

## Acknowledgement
The K-Radar dataset is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

We thank the maintainers of the following projects that enable us to develop `K-Radar`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`Rotated_IoU`](https://github.com/lilanxiao/Rotated_IoU) by lilanxiao, and [`kitti-object-eval-python`](https://github.com/traveller59/kitti-object-eval-python) by traveller59.

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
