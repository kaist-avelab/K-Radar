# Pre-processing
This is the documentation for Pre-processing of 4D Radar data with K-Radar dataset (refer to <a href="https://arxiv.org/abs/2303.06342">the paper</a>).
 

You can download the pre-processed 4D Radar data in google drive link: [URL](https://drive.google.com/drive/folders/1KwzaAx-k8sGx8k-0D7bFVooTkCfbXTfC?usp=drive_link)

<center><img src="/docs/imgs/preprocessing_density.png" width="90%"></center>

The density level of the pre-processed 4D Radar tensor data uploaded on Google Drive ranges from 0.01% to 10%. 
If you need data with different levels of density, please refer to the information below.

## First Stage
The first stage extracts the main measurements with high power from the 4D Radar tensor (e.g., percentile in RTNH and CFAR in common).

### Extract Measurement with Density

We provide pre-processing code in two versions: Cartesian coordinates and Polar coordinates.

<center><img src="/docs/imgs/preprocessing_stage1.png" width="90%"></center>


**1. Cartesian Coordinates**

In this case, use kradar detection version 1.1. Use radar_zyx_cube as input. 

Input data: 3DRT-XYZ

Output data: 4D Sparse Radar Tensor in cartesian coordinates

Modify data path and configs in `configs/sparse_rdr_data_generation/cfg_gen_wider_rtnh_10p.yml`. You can modify `QUANTILE_RATE`(line 34) to change the density of the data. 


```
$ python kradar_detection_v1_1.py
```

**2. Polar Coordinates**

In this case, use kradar detection version 2.1. Use rdr_polar_3d as input. 

Input data: 4DRT

Output data: 4D Sparse Radar Tensor in polar coordinates

Modify `dict_cfg` in `kradar_detection_v2_1.py`. You can modify `rate` in `def get_proportional_rdr_points` to change the density of the data. 

```
#in __main__
1. use the preprocessed data
dict_item = kradar_detection.get_proportional_rdr_points(dict_item)

2. save the preprocessed data
kradar_detection.save_proportional_rdr_pc()

#run code
$ python kradar_detection_v2_1.py
```

## Second Stage

<a href="https://drive.google.com/file/d/1hTHAHEYdhl6yEDdTz3uxv08vsdG4lxu7/view?usp=drive_link">Link</a>
Wide range, Quantile (RTNH) / 2 stage CFAR for Sidelobe filtering (RTNH+): code will be uploaded soon. (gif)
Function: datasetv1_1 generate~
Density: ref Enhanced
