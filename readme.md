<p align="center">
  <img src = "./docs/imgs/readme_logo.png" width="60%">
</p>

`KAIST-Radar (K-Radar)` (provided by ['AVELab'](http://ave.kaist.ac.kr/)) is a novel large-scale object detection dataset and benchmark that contains 35K frames of 4D Radar tensor (4DRT) data with power measurements along the Doppler, range, azimuth, and elevation dimensions, together with carefully annotated 3D bounding box labels of objects on the roads. K-Radar includes challenging driving conditions such as adverse weathers (fog, rain, and snow) on various road structures (urban, suburban roads, alleyways, and highways). In addition to the 4DRT, we provide auxiliary measurements from carefully calibrated high-resolution Lidars, surround stereo cameras, and RTK-GPS. This repository provides the `K-Radar` dataset, annotation tool for 3d bounding boxes, and the visualization tool for showing the inference results and calibrating the sensors.

![image](./docs/imgs/kradar_examples.png)

The URLs listed below are useful for using the K-Radar dataset and benchmark:
* <a href="https://arxiv.org/abs/2206.08171"> K-Radar paper and appendix </a>
* <a href="https://arxiv.org/abs/2303.06342"> 4D Radar object detection network of input with various density </a>
* <a href="https://arxiv.org/abs/2310.17659"> Enhanced network for 4D Radar using two-stage pre-processing </a>
* <a href="https://github.com/kaist-avelab/K-Radar_auto-labeling"> Repository of Auto-labeling for 4D Radar </a>
* <a href=""> Repository of Odometry for 4D Radar </a>
* <a href="https://www.youtube.com/watch?v=TZh5i2eLp1k"> The video clip that shows each sensor measurement dynamically changing during driving under the heavy snow condition </a>
* <a href="https://www.youtube.com/watch?v=ylG0USHCBpU"> The video clip that shows the 4D radar tensor & Lidar point cloud (LPC) calibration and annotation process </a>
* <a href="https://www.youtube.com/watch?v=ILlBJJpm4_4"> The video clip that shows the annotation process in the absence of LPC measurements of objects  </a>
* <a href="https://www.youtube.com/watch?v=U4qkaMSJOds"> The video clip that shows calibration results </a>
* <a href="https://www.youtube.com/watch?v=MrFPvO1ZjTY"> The video clip that shows the GUI-based program for visualization and neural network inference </a>
* <a href="https://www.youtube.com/watch?v=8mqxf58_ZAk"> The video clip that shows the information on tracking for multiple objects on the roads </a>

## License and Commercialization Inquiries
The `K-Radar` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.

The technologies in this repository have been developed by [`AVELab`](http://ave.kaist.ac.kr/) and are being commercialized by [`Zeta Mobility`](https://zetamobility-company.imweb.me/). For commercialization inquiries, please contact [`Zeta Mobility`](https://zetamobility-company.imweb.me/) (e-mail: zeta@zetamobility.com).

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

## Sensor Fusion

For the preparation and quantitative results of the camera, LiDAR, and 4D Radar sensor fusion, please refer to the following document: [Sensor Fusion Pipeline Guide](/docs/sensor_fusion.md).

<p align="center">
  <img src = "./docs/imgs/fusion1.gif" width="80%">
  <img src = "./docs/imgs/fusion2.gif" width="80%">
</p>

The images below showcase qualitative results, with yellow boxes representing detected objects from our sensor fusion AI (camera + LiDAR + 4D Radar), which we refer to as [Availability-aware Sensor Fusion (ASF)](https://arxiv.org/abs/2503.07029). From left to right, the images depict: (1) front-facing camera images, (2) LiDAR point clouds, (3) 4D Radar tensors, and (4) sensor attention maps (SAMs). ASF stands out with three key advantages: **sensor availability awareness** that maintains performance even when sensors fail or degrade in adverse conditions by automatically prioritizing available sensors (in the SAMs, red, green, and blue represent attention scores for Camera, LiDAR, and 4D Radar, with predominantly blue SAMs in adverse weather indicating 4D Radar automatically received the highest attention); **efficient computation structure** with minimal memory usage (1.5-1.6GB) and high processing speeds (13.5-20.5Hz); and **superior detection performance** (87.4% AP3D at IoU=0.3 and 73.6% AP3D at IoU=0.5) even including challenging weather conditions. For more details, please refer to [the paper](https://arxiv.org/abs/2503.07029).

## Detection & Tracking

For the preparation and quantitative results of the 4D Radar-based object detection, please refer to the following document: [Detection Pipeline Guide](/docs/detection.md).

The images below showcase qualitative results, with green boxes representing detected objects using 4D Radar. From left to right, the images depict (1) front-facing camera images (for reference), (2) LiDAR point clouds (for reference), and (3) 4D Radar tensors (input for the neural network).
Note that the camera images and LiDAR point clouds are shown for reference purposes only, and the bounding boxes from the 4D Radar-only detection are projected onto these visualizations.

<p align="center">
  <img src = "./docs/imgs/detection1.gif" width="80%">
  <img src = "./docs/imgs/detection2.gif" width="80%">
  <img src = "./docs/imgs/detection3.gif" width="80%">
</p>

For the preparation and quantitative results of the 4D Radar-based object tracking, please refer to the following document: [Tracking Pipeline Guide](/docs/tracking.md).

## Pre-processing

For the preparation and quantitative results of the 4D Radar-based object detection, please refer to the following document: [Pre-processing Pipeline Guide](/docs/preprocessing.md).

As shown in the figure below, the pre-processing of 4D Radar consists of two stages. The first stage extracts the main measurements with high power from the 4D Radar tensor (e.g., percentile in RTNH and CFAR in common), while the second stage removes noise from the first stage output. In the first stage, it is important to exclude peripheral measurements with low power and extract measurements with an appropriate density (typically 3~10%) to allow the detection network to recognize the shape of the object (refer to <a href="https://arxiv.org/abs/2303.06342">the paper</a>). However, extracting measurements at this density introduces noise called sidelobe, which interferes with precise localization of the object, as shown in the figure below. Therefore, pre-processing that indicates sidelobe to the network is essential, and applying this alone greatly improves detection performance, particularly the localization performance of objects (refer to <a href="https://arxiv.org/abs/2310.17659">the paper</a>). For quantitative results and pre-processed data, please refer to [the pre-processing document](/docs/preprocessing.md).

<p align="center">
  <img src = "./docs/imgs/preprocessing.gif" width="65%">
</p>

## Auto-labeling
This is the documentation for how to use our auto-labeling frameworks with the K-Radar dataset:

For the preparation and quantitative results of the 4D Radar-based object detection, please refer to the following document: [Auto-labeling Pipeline Guide](/docs/labeling.md).

The figure below shows the auto-labeling results for six road environments in the K-Radar dataset. Each cell, from left to right, displays the handmade labels, the detection results of the 4D Radar detection network trained with handmade labels (RTNH), and the network trained with automatically generated labels (i.e., auto-labels). Notably, the RTNH trained with auto-labels only from clear weather conditions performs robustly even in inclement weather. This implies that the distribution of 4D Radar data depends more on road conditions than weather conditions, allowing for the training of an all-weather resilient 4D Radar detection network using only clear weather auto-labels. For more details and quantitative results, refer to [the auto-labeling document](/docs/labeling.md).

<p align="center">
  <img src = "./docs/imgs/label_results.gif" width="80%">
</p>

Regarding the commercialization and the usage of auto-labeling technologies, please contact `Zeta Mobility`.

## Radar Synthesis
TODO

## Embedded Systems

Spiking Neural Network (Energy-efficient) & Hailo

## Odometry
We provide the location of a GPS antenna, essential for accurate ground-truth odometry. This location is precisely processed by integrating data from high-resolution LiDAR, RTK-GPS, and IMU data. To ensure the utmost accuracy, we verify the vehicle's location by correlating the LiDAR sensor data against a detailed, high-resolution map, as illustrated below. For security purposes, we present this location information in local coordinates rather than global coordinates (i.e., UTM). For more details, refer to [the odometry document](/docs/odometry.md).

<p align="center">
  <img src = "./docs/imgs/processed_odometry.png" width="80%">
</p>

## Acknowledgement
The K-Radar dataset is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_3&wr_id=27), and [Sangjae Cho](https://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=3), and advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

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
