# Uniqueness of K-Radar  
K-Radar is the **world’s first** and **largest** publicly available autonomous driving 4D radar dataset that provides full **RAED (Range–Azimuth–Elevation–Doppler)** tensor (4D tensor) data without any loss of dimensionality.  

The characteristics of existing 4D radar datasets are as follows:  
1) Most provide **sparse point cloud representations** pre-processed from raw tensor, which leads to performance degradation in 3D object detection due to sparsity [1, 2].  
2) Even when released in tensor form, as shown in the table below, the datasets were collected with real 4D radar sensors but only **partial RAED dimensions (2D or 3D tensor)** were included for public release [2, 7].  
3) To date, only **K-Radar [1] and RaDelft [7]** provide full 4D tensor across all RAED dimensions, with **K-Radar being the largest dataset** in terms of scale [5, 6].  

For points (1) and (2), it is assumed that most 4D radar hardware providers restricted external release of raw tensor data due to **technology protection and security policies**, which made it difficult to access complete 4D tensor.  

K-Radar overcame these technical and policy barriers by **publicly releasing full 4D tensor data for the first time**, enabling high-quality training for more accurate and robust object detection and tracking.  

---

## Available Public 4D Radar Tensor Datasets (as of 2025.09.09) [1,2,3,4,5,6,7].

| Dataset Name    | Year | Tensor Dim. | Radar Frames (total) | Key Features                  |
|-----------------|------|-------------|----------------------|-------------------------------|
| SCORP [3]       | 2020 | RAD (3D)    | 3.9K                 | No elevation                  |
| ColoRadar [4]   | 2022 | RAE (3D)    | 108K                 | No Doppler                    |
| Radatron [5]    | 2022 | RA (2D)     | 152K                 | No elevation, no Doppler      |
| RAIDal [6]      | 2022 | RAD (3D)    | 25K                  | No elevation                  |
| **K-Radar [1]** | **2022** | **RAED (4D)** | **35K**          | **World First & World Largest** |
| RaDelft [7]     | 2024 | RAED (4D)   | 16.9K                | –                             |

> 2D tensor: Radatron [5]  
> 3D tensor: SCORP [3], ColoRadar [4], RAIDal [6]  
> 4D tensor: K-Radar[1], RaDelft [7]  

---

- **K-Radar is the world’s first** dataset to provide full RAED tensor, and with **35K frames it represents the largest publicly available 4D tensor dataset (World Largest)**.   
- **According to Peng et al. [8]**, K-Radar includes the **widest range of adverse weather scenarios** among existing 4D radar datasets.  
- With these unique features, **K-Radar has been requested by more than 65 leading institutions worldwide** and has made significant contributions to advancing autonomous driving research with 4D radar.  

---

## References
[1] Paek, Dong-Hee, Seung-Hyun Kong, and Kevin Tirta Wijaya. "K-radar: 4d radar object detection for autonomous driving in various weather conditions." *Advances in Neural Information Processing Systems* 35 (2022): 3819-3829.  
[2] Yao, Shanliang, et al. "Exploring radar data representations in autonomous driving: A comprehensive review." *IEEE Transactions on Intelligent Transportation Systems* (2025).  
[3] Nowruzi, Farzan Erlik, et al. "Deep open space segmentation using automotive radar." *2020 IEEE MTT-S International Conference on Microwaves for Intelligent Mobility (ICMIM)*. IEEE, 2020.  
[4] Kramer, Andrew, et al. "Coloradar: The direct 3d millimeter wave radar dataset." *The International Journal of Robotics Research* 41.4 (2022): 351-360.  
[5] Madani, Sohrab, et al. "Radatron: Accurate detection using multi-resolution cascaded MIMO radar." *European Conference on Computer Vision*. Cham: Springer Nature Switzerland, 2022.  
[6] Rebut, Julien, et al. "Raw high-definition radar for multi-task learning." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2022.  
[7] Roldan, Ignacio, et al. "A deep automotive radar detector using the RaDelft dataset." *IEEE Transactions on Radar Systems* (2024).  
[8] Peng, Xiangyuan, et al. "4D mmWave Radar in Adverse Environments for Autonomous Driving: A Survey." *arXiv e-prints* (2025): arXiv-2503.  
