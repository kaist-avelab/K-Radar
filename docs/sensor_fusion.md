# Sensor Fusion

This is the documentation for how to use our sensor fusion frameworks with K-Radar dataset. We tested the K-Radar detection frameworks on the following environment:

* Python 3.8.13
* Ubuntu 18.04/20.04
* Torch 1.12.1+cu113
* CUDA 11.3
* opencv 4.2.0.32

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

3. Install PyTorch (We recommend pytorch 1.12.1.)

4. Install the dependencies
```
pip install -r requirements.txt
```

5. Build packages for Rotated IoU
```
cd ops
python setup.py develop
```

6. Unzip 'kradar_revised_label_v2_0.zip' in the 'tools/revise_label' directory.

7. Download pretrained weights of sensor-specific encoders in <a href="">this link</a> and put them all in the './pretrained' folder.

8. Download <a href="">pre-processed camera images</a>, <a href="">LiDAR point clouds</a>, and <a href="">4D Radar point clouds</a>.

9. Link the folders of pre-processed data in the config yaml file.

## Train & Evaluation
* To train the model, run
```
python main_train_0_args.py 
```

* To evaluate the model, run
```
python main_cond_0_args.py 
```

## Pretrained Weights and Logs

## Performance of ASF (4D Radar + LiDAR + Camera)
The reported values are ${AP_{BEV}^{IoU=0.3}}$ and ${AP_{3D}^{IoU=0.3}}$ for Sedan class. (based on the benchmark v1.0)

|Metric|Total|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BEV|88.6|88.2|90.2|98.9|89.0|80.4|89.2|78.4|
|3D|87.4|87.0|90.1|90.7|88.2|80.0|88.6|77.4|

The reported values are ${AP_{3D}^{IoU=0.3}$ and ${AP_{BEV}^{IoU=0.3}$ for Sedan and Bus and Truck classes. (based on the benchmark v2.0)

(1) ${AP_{3D}}$
| Class        | Total | Normal | Overcast | Fog  | Rain | Sleet | LightSnow | HeavySnow |
|:------------:|:-----:|:------:|:--------:|:----:|:----:|:-----:|:---------:|:---------:|
| Sedan        | 79.3  | 78.8   | 86.1     | 93.7 | 72.9 | 74.2  | 91.2      | 65.8      |
| Bus or Truck | 60.4  | 52.7   | 77.4     | -    | 8.0  | 69.2  | 87.9      | 69.1      |

(2) ${AP_{BEV}}$
| Class        | Total | Normal | Overcast | Fog  | Rain | Sleet | LightSnow | HeavySnow |
|:------------:|:-----:|:------:|:--------:|:----:|:----:|:-----:|:---------:|:---------:|
| Sedan        | 82.5  | 81.8   | 94.9     | 98.0 | 76.2 | 81.0  | 93.8      | 69.5      |
| Bus or Truck | 70.2  | 58.3   | 81.5     | -    | 8.0  | 72.2  | 96.1      | 89.6      |

Please refer to <href="">the logs<\a> for the performance of ASF with various sensor combinations (e.g., 4D Radar-only, 4D Radar + LiDAR) and strict metrics (i.e., IoU=0.5).

## Sensor Attention Map Visualization

## t-SNE Visualization
