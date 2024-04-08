# K-Radar Dataset Generation Process (from .bag files)

This is the documentation on the process of dataset generation in K-Radar

## Generating Lidar point clouds (LPCs), camera images, RTK-GPS, and IMU data from rosbag files
We have recorded the LPCs, camera images, RTK-GPS, and IMU data based on rosbag record function of ROS Melodic.

You can generate files LPCs, images, RTK-GPS, and IMU data in '.pcd', '.png', and '.txt' as follows:
```
# Build ros node for generation
cd K-Radar
cd dataset_utils/bag_decoder_ros
catkin_make

# Generating files
roscore
source ./devel/setup.bash
rosrun bag_decoder bag_decoder_node
```

Note that you should change required topics and path of bag files and folders in 'bag_decoder.cpp'.
