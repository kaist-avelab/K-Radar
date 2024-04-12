<p align="center">
  <img src = "./docs/imgs/readme_logo.png" width="60%">
</p>

`KAIST-Radar (K-Radar)` (provided by ['AVELab'](http://ave.kaist.ac.kr/)) is a novel large-scale object detection dataset and benchmark that contains 35K frames of 4D Radar tensor (4DRT) data with power measurements along the Doppler, range, azimuth, and elevation dimensions, together with carefully annotated 3D bounding box labels of objects on the roads. K-Radar includes challenging driving conditions such as adverse weathers (fog, rain, and snow) on various road structures (urban, suburban roads, alleyways, and highways). In addition to the 4DRT, we provide auxiliary measurements from carefully calibrated high-resolution Lidars, surround stereo cameras, and RTK-GPS. This repository provides the `K-Radar` dataset, annotation tool for 3d bounding boxes, and the visualization tool for showing the inference results and calibrating the sensors.

![image](./docs/imgs/kradar_examples.png)

The URLs listed below are useful for using the K-Radar dataset and benchmark:
* <a href="https://arxiv.org/abs/2206.08171"> K-Radar paper and appendix </a>
* <a href="https://ieeexplore.ieee.org/document/10186820"> Radar object detection network of input with various density </a>
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

The technologies in this repository have been developed by [`AVELab`](http://ave.kaist.ac.kr/) and are being commercialized by `Zeta Mobility`. For commercialization inquiries, please contact `Zeta Mobility` (e-mail: zeta@zetamobility.com).

<p align="center">
  <img src = "./docs/imgs/zeta_mobility.png" width="60%">
</p>

# K-Radar Dataset
For the preparation of the dataset and pipeline, please refer to the following document: [Dataset Preparation Guide](/docs/dataset.md).

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

For the preparation and quantitative results of the 4D Radar-based object detection, please refer to the following document: [Detection Pipeline Guide](/docs/detection.md).

The images below showcase qualitative results, with green boxes representing detected objects using 4D Radar. From left to right, the images depict (1) front-facing camera images (for reference), (2) LiDAR point clouds (for reference), and (3) 4D Radar tensors (input for the neural network).
Note that the camera images and LiDAR point clouds are shown for reference purposes only, and the bounding boxes from the 4D Radar-only detection are projected onto these visualizations.

<p align="center">
  <img src = "./docs/imgs/detection1.gif" width="80%">
  <img src = "./docs/imgs/detection2.gif" width="80%">
  <img src = "./docs/imgs/detection3.gif" width="80%">
</p>

## Pre-processing
This is the documentation for how to use our auto-labeling frameworks with the K-Radar dataset:

For the preparation and quantitative results of the 4D Radar-based object detection, please refer to the following document: [Pre-processing Pipeline Guide](/docs/preprocessing.md).

The images below showcase importance of preprocessing of 4D Radar. (TODO)

<p align="center">
  <img src = "./docs/imgs/preprocessing.gif" width="70%">
</p>

## Auto-labeling
This is the documentation for how to use our auto-labeling frameworks with the K-Radar dataset:

For the preparation and quantitative results of the 4D Radar-based object detection, please refer to the following document: [Auto-labeling Pipeline Guide](/docs/labeling.md).

<p align="center">
  <img src = "./docs/imgs/label_results.gif" width="80%">
</p>

Regarding the commercialization and the usage of auto-labeling technologies, please contact `Zeta Mobility`.

## Odometry
We provide the location of a GPS antenna, essential for accurate ground-truth odometry. This location is precisely processed by integrating data from high-resolution LiDAR, RTK-GPS, and IMU data. To ensure the utmost accuracy, we verify the vehicle's location by correlating the LiDAR sensor data against a detailed, high-resolution map, as illustrated below. For security purposes, we present this location information in local coordinates rather than global coordinates (i.e., UTM). The data of Sequence 1 is accessible in the `resources/odometry` directory. The repository for the odometry will be made available within the next few weeks.

![image](./docs/imgs/processed_odometry.png)

## Acknowledgement
The K-Radar dataset is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_3&wr_id=27), [Sangjae Cho](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=3), and [Hong-Woo Seok](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=25), and advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

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
