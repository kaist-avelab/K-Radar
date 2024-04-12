# Auto-labeling

This is the documentation for how to use our auto-labeling with K-Radar dataset. We tested the auto-labeling on the following environment:
* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* CUDA 11.3
* Torch 1.11.0+cu113
* opencv 4.2.0.32
* open3d 0.15.2

All codes are available in <a href="https://github.com/kaist-avelab/K-Radar_auto-labeling">the repository of auto-labeling for 4D Radar</a>.

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
pip install tqdm
pip install shapely
pip install SharedArray

###(if numpy >= 1.24)###
pip uninstall numpy 
pip install numpy==1.23.0
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

## Directory Structure
```
K-Radar_auto_labeling_project
      ├── configs
      ├── datasets
      ├── logs
      ├── models
      ├── ops (from K-Radar repository)
      ├── pipelines
      ├── project
            ├── auto-label
                  ├── auto-labels (download from below URL)
                  ├── configs
                  ├── datasets
                  ├── LODN_model_log (download from below URL)
                  ├── logs
                  ├── pipelines
                  ├── plt_save_folder
                  ├── RTNH_model_log (download from below URL)
                  ├── utils   
      ├── resources (from K-Radar repository)
      ├── tools (from K-Radar repository)
      ├── uis
      ├── utils
```

Auto-labels resulting from generation and refinement (using PVRCNN++) URL: <a href="https://drive.google.com/file/d/1D5Y90S1ib-68r1IKsqt7R7PglGwjwSgr/view?usp=drive_link">auto-labels</a>

Pretrained LiDAR object detection model download URL: <a href="https://drive.google.com/file/d/1wckO9tPC75chCGrQL7al-pCmTBmBYyQ5/view?usp=drive_link">LODN_model_log</a>

Pretrained 4D Radar object detection model download URL: <a href="https://drive.google.com/file/d/1xBEC2zNQVwtOv8aBqslnKQVbUE-LXlki/view?usp=drive_link">RTNH_model_log</a>

## Auto-label Generation
Generate auto-labels of 4D Radar data by run the LiDAR object detection networks. We use `PVRCNN++` and `SECOND` as the LiDAR object detection networks. We utilize the output from a 0.3 confidence score threshold.

### Modify Config
1. Modify `generated_label_name` in `main_quto_labels_generation.py` to your path for save results. (line 40)
2. Modify `list_dir_kradar` in `configs/cfg_PVRCNNPP_cond.yml` to your path which contains K-Radar datasets. (line 34)

### Run 

in `/project/auto_label`
```
$ python main_auto_labels_generation.py
```

## Auto-label Refinement
Refine the generated auto-labels using heuristic algorithm.

### Modify Config
1. Modify `pvrcnn_label_path` in `main_auto_labels_refinement.py` to your path which contains auto-label generation results. (line 31)
2. Modify `revice_pvrcnn_label_path` in `main_auto_labels_refinement.py` to your path for save refinement results. (line 35)

### Run
in `/project/auto_label`
```
$ python main_auto_labels_refinement.py
```

## Auto-label Validation
Validate generated and refined auto-label using hand-label data. 

### Modify Config
1. Modify `pvrcnn_label_path` in `main_auto_labels_validation.py` to your path which contains auto-label results. (line 37)
2. Modify the list of `all` in `main_auto_labels_validation.py` if you want to validate conditionally. (line 33)

### Run
in `/project/auto_label`
```
$ python main_auto_labels_validation.py
```

## Train Object Detection Network
Train object detection network.
* original RTNH: Radar object detection network using hand label
* RTNH-PVRCNN: Radar object detection network using auto-label
* PVRCNN++: LiDAR object detection network
* SECOND: LiDAR object detection network


The dataset can be divided into subsets based on weather conditions, categorized as normal, overcast, rain, sleet, fog, light snow, and heavy snow, allowing for training on each specific environment.

### Modify Config
1. Choose a detection model by modifying `PATH_CONFIG` in `main_train_cond.py`. (line 20)
2. If you want to train a model using auto-label, modify `label_version` and `revised_label_v_P` in `configs/cfg_RTNH_wide_cond.yml` to a path which contains auto-label results. (line 40, 41)
3. Modify `list_dir_kradar` in `configs/cfg_RTNH_wide_cond.yml` to your path which contains K-Radar datasets. (line 34)
4. Modify `rdr_spars[‘dir’]` in `configs/cfg_RTNH_wide_cond.yml` to your path which contains 4D Radar point cloud data. (line 65)
5. Modify `IS_DATA_SCENE_FILTERING` and `DATA_SCENE_FILTERING` in `configs/cfg_*_cond.yml` if you want to train model conditionally. (bottom of the file)

### Run
in `/project/auto_label`
```
$ python main_train_cond.py
```

The learning results are saved in the `/logs/exp_xxxxxx_xxxxxx`

## Test Object Detection Network
Test object detection network.
* original RTNH: Radar object detection network using hand label
* RTNH-PVRCNN: Radar object detection network using auto-label
* PVRCNN++: LiDAR object detection network
* SECOND: LiDAR object detection network

The dataset can be divided into subsets based on weather conditions, categorized as normal, overcast, rain, sleet, fog, light snow, and heavy snow, allowing for test on each specific environment.

### Modify Config
1. Choose a detection model by modifying `PATH_CONFIG` and `PATH_MODEL` in `main_test_cond.py.` (line 20, 24)
2. Modify `IS_DATA_SCENE_FILTERING` and `DATA_SCENE_FILTERING` in `configs/cfg_*_cond.yml` if you want to test model conditionally. (bottom of the file)

### Run
in `/project/auto_label`
```
$ python main_test_cond.py
```

## Visualize
Visualize detection networks inference results.
* RTNH: Radar object detection network
* PVRCNN++: LiDAR object detection network
* SECOND: LiDAR-based object detection network

It has two mode.
1. 3D Visualize
2. 2D BEV Visualize (using plt)

### Modify Config
1. Choose a detection model by modifying `PATH_CONFIG` and `PATH_MODEL` in `main_inf_vis_cond.py`. (line 32, 36)
2. Choose visualize mode by modifying `is_3D_vis` and `is_plt_vis` in
`main_inf_vis_cond.py`. (line 49, 55)

    2-1. If you want to save the results of plt, modify `is_plt_save` and `plt_save_path` in `main_inf_vis_cond.py`. (line 60, 61)

### Run
in `/project/auto_label`
```
$ python main_inf_vis_cond.py
```

## Model Zoo

### LiDAR Object Detection Network

(1) The reported values are ${AP}$ for `Sedan` class. (based on the hand label)
|Model|View|Overall|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SECOND|BEV|70.32|70.01|80.26|90.50|61.88|60.91|79.72|52.82|
|SECOND|3D |68.39|68.35|69.09|80.71|60.04|58.72|77.95|52.23|
|PVRCNN++|BEV|74.82|73.97|87.42|87.97|70.17|64.48|84.24|55.11|
|PVRCNN++|3D|68.47|67.63|78.39|87.10|68.14|59.36|82.52|50.05|

(2) The reported values are ${AP}$ for `Bus or Truck` class. (based on the hand label)
|Model|View|Overall|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SECOND|BEV|56.12|50.49|-|-|-|69.06|89.49|54.51|
|SECOND|3D |46.97|42.90|-|-|-|66.52|80.22|32.72|
|PVRCNN++|BEV|60.81|59.42|-|-|-|77.76|88.54|50.74|
|PVRCNN++|3D|57.55|57.75|-|-|-|76.74|83.80|37.88|

### 4D Radar Object Detection Network

* RTNH: Radar detection network (based on the hand label)
* RTNH-SECOND: Radar detection network (based on the auto-label from SECOND)
* RTNH-PVRCNN++: Radar detection network (based on the auto-label from PVRCNN++)

(1) The reported values are ${AP}$ for `Sedan` class. 
|Model|View|Overall|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTNH | BEV  | 56.7    | 53.8   | 68.3     | 89.6  | 49.3  | 55.6  | 69.4  | 60.3   |
| RTNH| 3D   | 48.2    | 45.5   | 58.8     | 79.3  | 40.3  | 48.1  | 65.6       | 52.6       |
| RTNH-SECOND    | BEV  | 54.04   | 51.88  | 67.80    | 88.66 | 41.42 | 34.42 | 68.26      | 51.11      |
|        RTNH-SECOND        | 3D   | 45.93   | 44.24  | 58.23    | 78.90 | 37.78 | 29.31 | 65.63      | 48.83      |
| RTNH-PVRCNN++  | BEV  | 54.65   | 47.08  | 69.38    | 89.17 | 48.51 | 36.49 | 67.52      | 52.02      |
|     RTNH-PVRCNN++    | 3D   | 46.30   | 44.28  | 60.09    | 78.36 | 39.42 | 28.60 | 63.05      | 50.40      |


(2) The reported values are ${AP}$ for `Bus or Truck` class. 
|Model|View|Overall|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTNH        | BEV  | 45.3    | 31.8   | -        | -   | -    | 34.4  | 89.3       | 78         |
|        RTNH     | 3D   | 34.4    | 25.3   | -        | -   | -    | 28.5  | 78.2       | 46.3       |
| RTNH-SECOND | BEV  | 35.32   | 26.30  | -        | -   | -    | 32.39 | 87.00      | 57.94      |
|      RTNH-SECOND       | 3D   | 27.98   | 24.08  | -        | -   | -    | 24.82 | 74.45      | 33.09      |
| RTNH-PVRCNN++ | BEV  | 44.08   | 31.76  | -        | -   | -    | 36.09 | 81.15      | 68.83      |
|     RTNH-PVRCNN++        | 3D   | 35.74   | 25.88  | -        | -   | -    | 28.63 | 76.56      | 53.79      |


### Conditionally trained 4D Radar Object Network 

* RTNH: Radar detection network (based on the hand label)
* RTNH-PVRCNN++: Radar detection network (based on the auto-label from PVRCNN++) with `all` condition
* RTNH-PVRCNN-NO: Radar detection network (based on the auto-label from PVRCNN++) with `Normal, Overcast` condition
* RTNH-PVRCNN-NOFRL: Radar detection network (based on the auto-label from PVRCNN++) with `Normal, Overcast, Fog, Rain, Light snow` condition

<center><img src="/docs/imgs/label_cond.png" width="450" height="300"></center>

(1) The reported values are ${AP}$ for `Sedan` class. 
|Model|View|Overall|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTNH             | bev  | 56.7    | 53.8   | 68.3     | 89.6  | 49.3  | 55.6  | 69.4       | 60.3       |
|      RTNH            | 3D   | 48.2    | 45.5   | 58.8     | 79.3  | 40.3  | 48.1  | 65.6       | 52.6       |
| RTNH-PVRCNN-NO   | bev  | 45.67   | 46.90  | 60.28    | 83.22 | 38.53 | 17.54 | 44.57      | 38.75      |
|      RTNH-PVRCNN-NO             | 3D   | 36.30   | 43.32  | 56.39    | 61.00 | 29.11 | 6.75  | 36.43      | 34.70      |
| RTNH-PVRCNN-NOFRL | bev  | 53.68   | 47.24  | 68.77    | 89.04 | 47.69 | 22.43 | 65.36      | 49.32      |
|     RTNH-PVRCNN-NOFRL              | 3D   | 45.06   | 44.14  | 59.65    | 78.31 | 38.73 | 10.09 | 59.78      | 42.24      |
| RTNH-PVRCNN++      | bev  | 54.65   | 47.08  | 69.38    | 89.17 | 48.51 | 36.49 | 67.52      | 52.02      |
|    RTNH-PVRCNN++               | 3D   | 46.30   | 44.28  | 60.09    | 78.36 | 39.42 | 28.60 | 63.05      | 50.40      |

(2) The reported values are ${AP}$ for `Bus or Truck` class. 
|Model|View|Overall|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTNH           | bev  | 45.3    | 31.8   | 32       | 0   | 0    | 34.4  | 89.3       | 78         |
|    RTNH            | 3D   | 34.4    | 25.3   | 31.1     | 0   | 0    | 28.5  | 78.2       | 46.3       |
| RTNH-PVRCNN-NO | bev  | 16.41   | 31.45  | 29.81    | 0   | 0    | 3.03  | 11.87      | 0.00       |
|   RTNH-PVRCNN-NO             | 3D   | 14.62   | 27.96  | 27.59    | 0   | 0    | 1.52  | 8.82       | 0.00       |
| RTNH-PVRCNN-NOFRL   | bev  | 24.14   | 29.02  | 30.72    | 0   | 0    | 4.81  | 88.44      | 0.00       |
|    RTNH-PVRCNN-NOFRL       | 3D   | 18.49   | 22.93  | 30.72    | 0   | 0    | 3.03  | 75.48      | 0.00       |
| RTNH-PVRCNN++    | bev  | 44.08   | 31.76  | 30.84    | 0   | 0    | 36.09 | 81.15      | 68.83      |
|    RTNH-PVRCNN++            | 3D   | 35.74   | 25.88  | 25.54    | 0   | 0    | 28.63 | 76.56      | 53.79      |
