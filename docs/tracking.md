# Tracking

This is the documentation for how to use our tracking frameworks with K-Radar dataset. We tested the K-Radar tracking frameworks on the following environment:

* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* CUDA 11.3
* Torch 1.11.0+cu113
* opencv 4.2.0.32
* open3d 0.15.2

For more details, please refer to [the paper](https://arxiv.org/abs/2502.01357).

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

9. nuscenes-devkit

We use the operations from <a href="https://github.com/open-mmlab/OpenPCDet">OpenPCDet</a> repository and acknowledge that all code in `ops` directory is sourced from there.
To align with our project requirements, we have made several modifications to the original code and have uploaded the revised versions to our repository.
We extend our gratitude to MMLab for their great work.

## Directories

The code and pre-processed data for Bayes-4DRTrack are available as `code_processed_data.zip` in [this URL](https://drive.google.com/drive/folders/1H993jlQPXehxBhag9e7HQTlVYDLQ4Kpt?usp=drive_link).

```
Bayes-4DRTrack # MOT
    ├── Bayes_4DRTrack_log 
    ├── configs
    ├── data_loader
    ├── docs
    ├── kradar_detection_result
    ├── mot_3d
    ├── mot_tools
    ├── prediction_tools
    ├── preprocessing
    ├── pretrained
    ├── raw_data_preprocessing
K-Radar # Object Detection
    ├── configs
    ├── ...
    ├── main_cond_0_for_tracking_mc_dropout.py # for preprocess detection results
```

## Multi-Object Tracking

### Modify Config
If you want run mot using detection results with MC Dropout and Loss attenuation, 
modify `kradar_baseline` to `kradar_mc_atte`.

### Run
```
$ python mot_tools/main_kradar.py --name Bayes-4DRTrack10Hz --det_name RTNH --config_path ./configs/kradar_configs/mdis.yaml --result_folder ./kradar_mot_results_server --data_folder ./kradar_detection_result/kradar_baseline/data_dir_10hz_kradar
```

### Result transform to nuscene format for evaluation
```
$ python mot_tools/nuscenes_result_creation_kradar.py --name Bayes-4DRTrack10Hz --result_folder ./kradar_mot_results_server/ --data_folder ./kradar_detection_result/kradar_baseline/data_dir_10hz_kradar/
```

### Evaluation
```
python mot_3d/utils/evaluate_kradar.py --result_path ./kradar_mot_results_server/Bayes-4DRTrack10Hz/results/car/results.json --output_dir ./kradar_mot_results_server/Bayes-4DRTrack10Hz/results --dataroot ./kradar_detection_result/kradar_baseline/raw_data
```

## Motion Prediction 

### Create trajectory label for motion prediction network
```
$ python -m prediction_tools.track_processor_kradar
```
then, the output is saved at './mot_3d/prediction/prediction_data/[name]'.

You can modify the configure in './prediction_tools/track_processor_kradar.py'.

### Train and test the motion prediction network
```
$ python -m prediction_tools.train_motion_prediction_kradar
```
You can modify the configure in './mot_3d/prediction/configs/cfg_basic_transformer_kradar.py'.

### MC Dropout predcition results visualize
```
$ python -m prediction_tools.vis_bnn_result_demo
```

## Object Detection

### Modify Config
You can modify config in '.../K-Radar/configs/cfg_RTNH_wide_tracking.yml'

### Run
```
$ python .../K-Radar/main_cond_0_for_tracking_mc_dropout.py 
```

## Preprocessing

### Detection
In this part, we add tracking token to K-Radar detection results and get Doppler value.

1. Create detection results
    ```
    $ python .../K-Radar/main_cond_0_for_tracking_mc_dropout.py 
    ```

2. Change log name in '.../K-Radar/logs/'

3. Get Doppler value

    - input: prediction for tracking
        
    - output: prediction for tracking with Doppler
    ```
    $ python .../Bayes-4DRTrack/raw_data_preprocessing/detection_preprocessing/main_get_doppler_points_for_rdr_sparse.py

    $ python .../Bayes-4DRTrack/raw_data_preprocessing/detection_preprocessing/detection_token_revise.py
    ```

4. Transform coordinates from local to global frame

    - input: prediction for tracking in local coordinate

    - output: prediction for tracking in global coordinate

    ```
    $ python .../Bayes-4DRTrack/raw_data_preprocessing/detection_preprocessing/local2global.py
    ```

5. Convert the result to NuScenes format

    - input: kitti format (txt)

    - output: nuscenes format (json)
    ```
    $ python .../Bayes-4DRTrack/raw_data_preprocessing/detection_preprocessing/revise_detection2nusences.py
    ```

### Tracking input

In this part, we make input of tracking network using detection results.

```
$ bash .../Bayes-4DRTrack/preprocessing/kradar_data/kradar_preprocess.sh
```

## Model Zoo

### Performance comparison with other 3D MOT systems
| Method                 | AMOTA↑ | AMOTP↓ | TP↑   | FP↓   | FN↓   | IDS↓ |
|------------------------|--------|--------|-------|-------|-------|------|
| AB3DMOT                | 0.339  | 1.281  | 7778  | **1866** | 10857 | 437  |
| Probabilistic 3D MOT   | 0.485  | 1.069  | 9930  | 2329  | 8893  | 259  |
| SimpleTrack            | 0.497  | 1.027  | 10359 | 2651  | 8428  | 295  |
| **Ours** | **0.543** | **0.865** | **11711** | 3374  | **7114** | **258** |

### Performance comparison of various bayesian approximation (V denotes using estimated variance)
| Prediction              | Detection                         | AMOTA↑ | AMOTP↓ | TP↑   | FP↓   | FN↓   | IDS↓ | Bayes_4DRTrack_log |
|------------------------|-----------------------------------|--------|--------|-------|-------|-------|------|---|
| Baseline (CV)          | RTNH                              | 0.492  | 1.043  | 10373 | 2850  | 8403  | 315  | Base_Base |
| MC Dropout (Pred)      | RTNH                              | 0.494  | 1.006  | 10351 | 2774  | 8395  | 306  | MC_Base |
| MC Dropout (Pred) + V  | RTNH                              | 0.495  | 1.000  | 10364 | 2774  | 8389  | 299  | MCV_Base |
| Baseline (CV)          | Loss Attenuation                  | 0.515  | 1.011  | 10354 | 2501  | 8538  | 191  | Base_Att |
| MC Dropout (Pred) + V  | Loss Attenuation                  | 0.521  | 0.937  | 10371 | **2483**  | 8542  | **170** | MCV_Att |
| Baseline (CV)          | MC Dropout + Loss Attenuation     | 0.537  | 0.971  | 11223 | 2969  | 7595  | 265  | Base_MCAtt |
| Baseline (CV)          | MC Dropout + Loss Attenuation + V | 0.529  | 1.005  | 11255 | 3175  | 7545  | 283  | Base_MCAttV |
| MC Dropout (Pred) + V  | MC Dropout + Loss Attenuation     | **0.543**  | **0.865**  | **11711** | 3374  | **7114** | 258 | MCV_MCAtt |
| MC Dropout (Pred) + V  | MC Dropout + Loss Attenuation + V | 0.535  | 0.925  | 11236 | 2961 | 7596  | 251  | MCV_MCAttV |

### Logs and pretrained pth

All the codes, logs, and pretrained pth files are available at [this URL](https://drive.google.com/drive/folders/1H993jlQPXehxBhag9e7HQTlVYDLQ4Kpt?usp=drive_link):

- Bayes-4DRTrack

    Other 3D_MOT_log: 3D_MOT_log.zip

    Bayes_4DRTrack_log: Bayes_4DRTrack_log.zip

    kradar_detection_result: kradar_detection_result.zip

    kradar_mot_results_server: kradar_mot_results_server.zip

    pretrained: Bayes-4DRTrack.pretrained.zip

- K-Radar

    pretrained URL: K-Radar.pretrained.zip
