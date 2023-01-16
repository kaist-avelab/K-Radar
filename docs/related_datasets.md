### Table A. Additional information of K-Radar and related datasets. HR and LR refer to High Resolution Lidar with more than 64 channels and Low Resolution with less than 32 channels, respectively. Bbox., Tr.ID, and Odom. refer to bounding box annotation, tracking ID, and odometry, respectively. ‘-’, and ‘n/a’ denote ‘not relevant’ and ‘not applicable’, respectively.

| Dataset | Radar tensor | Radar point cloud | Lidar point cloud | Camera | GPS | Bbox. | Tr. ID | Odom. | Weather conditions | Time | Num. labelled data | Num. labelled train data | Num. labelled val. data | Num. labelled test data | Num. Radar data | Num. Lidar data | Num. camera data | Num. 3D bboxes | Num. 2D bboxes | Num. points of objects | Road type | Driving period [hour] | Maximum range of Radar [m] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| K-Radar (ours) | 4D | 4D | HR. | 360. | RTK | 3D | O | O | overcast, fog, rain, sleet, snow | d/n | 35K | 17.5K | - | 17.5K | 38.9K | 37.7K | 112K | 93K | - | - | urban, highway, alleyway, suburban, university, mountain, parkinglots, shoulder | 1 | 118 |
| VoD | X | 4D | HR. | Front | RTK | 3D | O | O | X | day | 8.7K | 5.1K | 1.3K | 2.3K | n/a | n/a | n/a | 123K | - | - | urban | 0.2 | 64 |
| Astyx | X | 4D | LR. | Front | X | 3D | X | X | X | day | 0.5K | 0.4K | - | 0.1K | n/a | n/a | n/a | 3K | - | - | urban | 0.01 | 100 |
| RADDet | 3D | 3D | X | Front | X | 2D | X | X | X | day | 10K | 8K | - | 2K | n/a | n/a | n/a | - | 2.8K | - | n/a | n/a | 50 |
| Zendar | 3D | 3D | LR. | Front | GPS | 2D | O | O | X | day | 4.8K | n/a | - | n/a | n/a | n/a | n/a | - | 11.3K | - | urban | n/a | 90 |
| RADIATE | 3D | 3D | LR. | Front | GPS | 2D | O | O | overcast, fog, rain, snow | d/n | 44K | 33K | - | 11K | n/a | n/a | n/a | - | 200K | - | urban, highway, parkinglots, suburban | 3 | 100 |
| CARRADA | 3D | 3D | X | Front | X | 2D | O | X | X | day | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - | 7K | - | parkinglots | 0.35 | 50 |
| CRUW | 3D | 3D | X | Front | X | Point | O | X | X | day | n/a | n/a | n/a | n/a | 396K | - | n/a | - | - | 260K | urban, highway, parkinglots, suburban | 3.5 | n/a |
| NuScenes | X | 3D | LR. | 360. | RTK | 3D | O | O | overcast, rain | d/n | 40K | 28K | 6K | 6K | n/a | n/a | n/a | 1.4M | - | - | urban, highway suburban | 5.5 | n/a |
| Waymo | X | X | HR. | 360. | X | 3D | O | X | overcast | d/n | 230K | 160K | 40K | 30K | - | n/a | n/a | 12M | - | - | urban, suburban | 6.4 | - |
| KITTI | X | X | HR. | Front | RTK | 3D | O | O | X | day | 15K | 7.5K | - | 7.5K | - | n/a | n/a | 80K | - | - | suburban, highway | 1.5 | - |
| BDD100k | X | X | X | Front | RTK | 2D | O | O | overcast, fog, rain, snow | d/n | n/a | n/a | n/a | n/a | - | - | 120M | - | 3.3M | - | urban, highway, parkinglots | 1.1K | - |
