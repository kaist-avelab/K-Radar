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

<a href="https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_?usp=share_link"> Download link </a>

After downloading all the dataset, please arrange the sequence directory with the following structure:
```
├── sequence_number (i.e., 1, 2, ..., 58)
      ├── cam_front
      ├── cam_left
      ├── cam_rear
      ├── cam_right
      ├── description.txt
      ├── info_calib
      ├── info_label
      ├── os1-128
      ├── os2-64
      ├── radar_zyx_cube
```

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
conda create -n kradar python=3.8.13 -y
conda activate kradar
```

3. Install PyTorch (We recommend pytorch 1.11.0.)

4. Install the dependencies
```
pip install -r requirements.txt
```

5. Build packages for kradar
```
cd utils/Rotated_IoU/cuda_op
python setup.py install
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

## Revising K-Radar Label

There are two primary revisions to our K-Radar label:

1. **Quality Enhancement**: The quality of the K-Radar label has been significantly improved. For instance, the size of the bounding boxes is now better suited to individual objects. Missing labels have been updated using camera images and LiDAR point cloud data. Furthermore, the tracking IDs of the objects have been amended. This revision was made possible by the Information Processing Lab, as acknowledged. We refer to this version of the updated label as v2.0. (The previous one is the label v1.0) You can download it using [the provided link](https://drive.google.com/file/d/1hEN7vdqoLaHZFf7woj5nj4hjpP88f7gU/view?usp=sharing).

2. **Visibility Discrimination**: We now categorize the visibility of each object based on its appearance in LiDAR and 4D Radar.

![image](./docs/imgs/revised_label.png)
An example of a label of an object invisible to Radar but visible to LiDAR illustrated in gray dashed boxes in (a) the front view image, (b) LiDAR point cloud, and (c) 4D Radar heatmap.

Most 4D Radar object detection datasets use LiDAR-detected 3D bounding boxes as reference labels. However, a key difference between LiDAR and 4D Radar sensors is their respective installation locations. In detail, while LiDAR is typically installed on the roof of a vehicle, 4D Radar is placed in front of or directly above the vehicle bumper, as illustrated by the dashed green box in Figure above (a). As a result of this sensor placement disparity, such objects that are detected by LiDAR are not be visible to 4D Radar as shown in Figure above (b,c). This issue leads to the presence of labels that are invisible to Radar in 4D Radar datasets, but visible in LiDAR measurements. Such inconsistencies can negatively impact the performance of 4D Radar object detection neural networks, resulting in a number of false alarms. To address this issue, we distinguish labels that are invisible to Radar and we demonstrate this revision can improve the performance of the 4D Radar object detection network as shown in Model Zoo. We refer to this version of the updated label as v2.1. You can download it using the provided link.

## Sending 4D raw data via shipping

Given the considerable size of our research dataset, distributing it via Google Drive has proven difficult. Instead, we've been making it available through our local server. However, we're aware of the transfer speed limitations of our server and are seeking an alternative method.

If you're able to provide us with a hard drive of 16TB capacity or larger, we can directly transfer the raw data to it and send it to your institution. The method of provision is flexible: you could opt for international shipping, make a purchase via Amazon, or consider other avenues.

It's important to emphasize that we're offering this service on a **non-profit** basis. Several esteemed research institutions (like Washington Univ., KAIST, NYU, and National Yang Ming Chiao Tung Univ.) as well as companies (like Motional and GM) have previously received data using this method.

## Model Zoo
The reported values are ${AP_{3D}}$ for `Sedan` class. (based on the label v1.0)
|Name|Total|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|Pretrained|Logs|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|RTNH|47.4|49.9|56.7|52.8|42.0|41.5|50.6|44.5|<a href="https://drive.google.com/drive/folders/1Z9f0yiddLKkygkg-QKBsrCDK2OvNWGIA?usp=drive_link">Link</a>|<a href="https://drive.google.com/file/d/1b4DTwO_SnKOfPrn1F6UQZ-dy4Gi91Eun/view?usp=drive_link">Link</a>|
|RTN|40.1|45.6|48.8|46.9|32.9|22.6|36.8|36.8|<a href="https://drive.google.com/drive/folders/1Z9f0yiddLKkygkg-QKBsrCDK2OvNWGIA?usp=drive_link">Link</a>|<a href="https://drive.google.com/file/d/1b4DTwO_SnKOfPrn1F6UQZ-dy4Gi91Eun/view?usp=drive_link">Link</a>|
|PILLARS|45.4|52.3|61.0|42.2|44.5|22.7|40.6|29.7|<a href="https://drive.google.com/drive/folders/1Z9f0yiddLKkygkg-QKBsrCDK2OvNWGIA?usp=drive_link">Link</a>|<a href="https://drive.google.com/file/d/1b4DTwO_SnKOfPrn1F6UQZ-dy4Gi91Eun/view?usp=drive_link">Link</a>|

Based on our findings presented, we elucidate the capability of the 4D Radar in terms of 3D object detection and its resilience in the face of adverse weather conditions. This is evident from its relatively stable performance under conditions of sleet and heavy snow, especially when compared to the performance metrics of LiDAR neural networks. We do not claim that 4D Radar is always superior to LiDAR in 3D object detection. In other words, we note that the performance of LiDAR neural networks could surpass that of the 4D Radar neural network with improvements in its structure or hyper-parameters.

## Odometry
TODO

## License
The `K-Radar` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.

## Acknowledgement
The K-Radar dataset is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

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
