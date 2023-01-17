%% Load files
pathBaseDir = 'E:\radar_bin_lidar_bag_files\';
cellNameFiles = ...
    {'21_11_30_01', ...
     '21_11_30_03', ...
     };

nameFolderRadarBin = 'radar_bin_files\';
nameTailRadarBin = '.bin';
nameFolderLidarBag = 'lidar_bag_files\';
nameTailLidarBag = '.bag';

load('info_arr.mat') % arrRange, arrAzimuth, arrElevation
RAD2DEG = 180/pi;
DEG2RAD = pi/180;
arrAzimuthRad = arrAzimuth.*DEG2RAD;
arrElevationRad = arrElevation.*DEG2RAD;

%%% Hyperparams %%%
x_min = 0; % [m]
x_per_bin = 0.4; % [m]
x_max = 100-x_per_bin; % [m]

y_min = -80; % [m]; -x_max*sin(53*pi/180)
y_per_bin = 0.4; % [m]
y_max = 80-y_per_bin; % [m]; x_max*sin(53*pi/180)

z_min = -30; % [m]; -x_max*sin(18*pi/180) = -30.9
z_per_bin = 0.4; % [m]
z_max = 30-z_per_bin; % [m]; x_max*sin*18*pi/180) = 30.9

arr_x = x_min:x_per_bin:x_max;
arr_y = y_min:y_per_bin:y_max;
arr_z = z_min:z_per_bin:z_max;

len_x = length(arr_x);
len_y = length(arr_y);
len_z = length(arr_z);
%%% Hyperparams %%%

% idxFile = 4; % 'static_1'
