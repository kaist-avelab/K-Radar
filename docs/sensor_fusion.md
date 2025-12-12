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

8. Download K-Radar common dataset, pre-processed camera images, LiDAR point clouds, and 4D Radar point clouds.

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

## Pretrained weights and Logs

## Performance
The reported values are ${AP_{3D}^{@0.3}$ and ${AP_{BEV}^{@0.3}$ for Sedan class. (based on the label v1.0)
|Name|Total|Normal|Overcast|Fog|Rain|Sleet|LightSnow|HeavySnow|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|${AP_{3D}^{@0.3}$|47.4|49.9|56.7|52.8|42.0|41.5|50.6|44.5|
|${AP_{BEV}^{@0.3}$|40.1|45.6|48.8|46.9|32.9|22.6|36.8|36.8|

The reported values are ${AP_{3D}^{@0.3}$ and ${AP_{BEV}^{@0.3}$ for Sedan and Bus and Truck classes. (based on the label v2.0)
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
