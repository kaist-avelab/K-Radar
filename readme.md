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
      ├── dataset_utils
      ├── configs
      ├── datasets
      ├── models
      ├── pipelines
      ├── resources
      ├── uis
      ├── utils
      ├── logs
├── kradar_dataset
      ├── kradar_0
            ├── 1
            ├── 2
            ...
      ├── kradar_1
            ...
            ├── 57
            ├── 58
```

* Via Google Drive Urls

The `K-Radar` dataset is being uploaded to Google Drive. It will take time to upload our entire dataset.
The 4D Radar tensor is only available on our server because it is too large to upload to Google Drive. (We can charge for up to 2TB of Google Drive storage space.)
We provide camera images, Lidar point cloud, RTK-GPS, and Radar tensor (for network input) via Google Drive for flexibility.

1. Sequence 1 - <a href="https://drive.google.com/drive/folders/1nOZ6kbJYpevn9qKHDA4kLNtGUgyd9UHe?usp=share_link" title="seq1">link1</a>, <a href="https://drive.google.com/drive/folders/1vAJsPcMjon8fJCPfDmb0LosdtmYP51kE?usp=share_link" title="seq1">link2</a>, <a href="https://drive.google.com/drive/folders/1qFoy0GsZ0Pbp2_IBNlBQbBPT0ln2O77o?usp=share_link" title="seq1">link3</a>, <a href="https://drive.google.com/drive/folders/10BMtMDFH-qTGjwtP7I3NSmSqISuS6ST0?usp=share_link" title="seq1">link4</a>

2. Sequence 2 - <a href="https://drive.google.com/drive/folders/1jwDrkCesbnMCPtqSl12GKlnkb--NKcxN?usp=sharing_link" title="seq2">link1</a>, <a href="https://drive.google.com/drive/folders/1SBPnsVNzRYq2PQzWOivJXmjQjLBwJ_0l?usp=sharing_link" title="seq2">link2</a>, <a href="https://drive.google.com/drive/folders/1BrMdurvFAx3Le49Br1pYzcCWDge2lfW0?usp=sharing_link" title="seq2">link3</a>, <a href="https://drive.google.com/drive/folders/1tgtw8owHFOgEuRNMTMaQVtfYryzg0a5Y?usp=share_link" title="seq2">link4</a>

3. (On going)

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
├── description.txt: 
├── info_calib:
├── info_label: 
├── info_label_rev: 
├── lidar_bev_image:
├── lidar_bev_image_origin:
├── lidar_point_cloud:
├── os1-128:
├── os2-64:
├── radar_bev_image:
├── radar_bev_image_origin:
├── radar_tesseract:
├── radar_zyx_cube:
├── time_info:
```

## Requirements

1. Clone the repository
```
git clone https://github.com/kaist-avelab/K-Radar.git
cd K-Radar
```

2. Create a conda environment
```
conda create -n kradar python=3.9
conda activate kradar
```

3. Install PyTorch

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
python train_gpu_0.py ...
```

* To evaluate the model, modifying the lines that we have noted and run
```
python eval_conditional_0.py ...
```

## Code Hierarchy
![image](./resources/imgs/code_hierarchy.png)

## Annotation process

1. [Data generation process] (from rosbag and radar ADC files)
2. [Calibration process]

## GUI-based devkits

1. [Visualization]
2. [Inference]

## License
The `K-Radar` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.

## Acknowledgement
The K-Radar dataset is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz/), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

We thank the maintainers of the following projects that enable us to develop `K-Radar`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`Rotated_IoU`](https://github.com/lilanxiao/Rotated_IoU) by lilanxiao, and [`kitti-object-eval-python`](https://github.com/traveller59/kitti-object-eval-python) by traveller59.

This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 01210790) and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2021R1A2C3008370).

## Citation

If you find this work is useful for your research, please consider citing:
```
@InProceedings{paek2022kradar,
  title     = {K-Radar: 4D Radar Object Detection Dataset and Benchmark for Autonomous Driving in Various Weather Conditions},
  author    = {Paek, Dong-Hee and Kong, Seung-Hyun and Wijaya, Kevin Tirta},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  month     = {December},
  year      = {2022},
  url       = {https://github.com/kaist-avelab/K-Radar}
}
```

