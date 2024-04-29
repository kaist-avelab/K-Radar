# Calibration 


This is the documentation about lidar-camera calibration. We offer the calibration parameters from lidar to camera across all sequences in the K-radar dataset. The calibration file can be found in <a herf="./수정필요"> lidar-camera calibration</a>. Note that we assume calibration parameters reamin consistent when driving on the same road in order to obtain certain calibration parameters when the camera is obstructed.

<p align="center">
    <img src="./img/calibration_01.png" width="80%">
</p>



## Requirements
1. Clone the repository
```
git clone https://github.com/kaist-avelab/K-Radar.git
cd K-Radar
```

2. Modify the code in <a herf="https://github.com/kaist-avelab/K-Radar/blob/main/uis/ui_calibration.py">**ui_calibration.py**</a>
```python
Revise line 38: loadpath = "" # Specify the diretory path to load the calibration file
Revise line 39: savepath = "" # Specify the diretory path to save the calibration file
```

3. Execute **main_calib.py**
```
python main_calib.py
```

4. Click the **Lidar-Camera Calibration** and select a sequence from the K-radar dataset.

5. Click one of four stereo cameras(total 8 cameras) such as **Front(L)**, **Front(R)**, etc.

6. Load the calibration parameters using **Load Params**

7. If you want to save the calibration parameters, click **Save Params**

## Calibration Parameters
The file extension for the calibration parameters is '.yml', and each piece of information signifies as the following:
```yml
# Camera number
cam_number: 1               

# Camera Intrinsic Parameters
fx: 567.720776478944        
fy: 577.2136917114258       
px: 628.720776478944        
py: 369.3068656921387       

# Distortion Parameters
k1: -0.028873818023371287   
k2: 0.0006023302214797655   
k3: 0.0039573086622276855   
k4: -0.005047176298643093   
k5: 0.0                     

# Extrinsic Parameters from liadar to camera
roll_ldr2cam: 89.6          
pitch_ldr2cam: 0.59         
yaw_ldr2cam: 88.6           
x_ldr2cam: 0.18                     
y_ldr2cam: -2.42861286636753e-17    
z_ldr2cam: -0.019999999999999997    
```

Additionally, the camerea number signifies as the following:
```yml
cam_number: 1 # left  part of front stereo
cam_number: 2 # right part of front stereo

cam_number: 3 # left  part of right stereo
cam_number: 4 # right part of right stereo

cam_number: 5 # left  part of rear stereo
cam_number: 6 # right part of rear stereo

cam_number: 7 # left  part of left stereo
cam_number: 8 # right part of left stereo
```

# Explanation for each parameters
TODO

