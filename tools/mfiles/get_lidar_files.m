%% Load files
clear, clc, close all

pathFolderLidarBag = 'E:\radar_lidar_file_1004\pcd_files\lidar_bag\static_';
pathSave = 'E:\sradar\RETINA4F_Exam_Protected\data_files\';

idxFile = 1; % '2021_09_08_20_29_'
frameIdx = 50; % frame index of radar bin in static case
idxDopplerOffset = 10;

%% Load point cloud from bag
path_rosbag = strcat(pathFolderLidarBag, num2str(idxFile), '.bag');
bag = rosbag(path_rosbag);
points = select(bag,'Topic','/os_cloud_node/points');
cam_img = select(bag,'Topic','/usb_cam/image_raw/compressed');
cell_points = readMessages(points);
cell_img = readMessages(cam_img);

%% time information
time_start = bag.StartTime;
time_end = bag.EndTime;
vpa(time_start)
vpa(time_end)

% us 단위
vpa(time_start)
time = int64(time_start*1e6);

%% Set point cloud index from bag
idx_bag = 1;
pt_1 = cell_points{idx_bag};
img_1 = cell_img{idx_bag};
xyz_1 = readXYZ(pt_1);
img = readImage(img_1);

%% Camera
path_img = strcat(pathSave, 'cam_', num2str(time), '.png');
path_img

imwrite(img, path_img)

%% Point cloud
path_pc = strcat(pathSave, 'pc_', num2str(time), '.pcd');
path_pc

% pt_1.Fields(4)
% pt_1.Fields(6)

intensity = readField(pt_1, 'intensity');
reflectivity = readField(pt_1, 'reflectivity');

ptCloud = pointCloud(xyz_1, 'Intensity', intensity);
pcwrite(ptCloud, path_pc)


%% Plot pointcloud
f = figure;

roi_x_min = 0;
roi_x_max = 50*0.6;
roi_y_min = -40*0.6;
roi_y_max = 40*0.6;
roi_z_min = -5;
roi_z_max = 3;

ptCloud = pointCloud(xyz_1);
roi = [roi_x_min roi_x_max ...
        roi_y_min roi_y_max ...
        roi_z_min roi_z_max];
indices = findPointsInROI(ptCloud, roi);
ptCloud = select(ptCloud, indices);

pcshow(ptCloud, 'BackgroundColor', [1 1 1], 'MarkerSize', 15)
view(2)
xlim([roi_x_min, roi_x_max])
ylim([roi_y_min, roi_y_max])
% grid off
% axis off

grid on

hgexport(f(1), sprintf('temp'), hgexport('factorystyle'), 'Format', 'png','Resolution',500,'ShowUI','off');

% saveas(f, 'temp.png')

%% Figure 불러오기
img = imread('temp.png');
[r, c, channel] = size(img);

r_cen = round(r/2);
c_cen = round(c/2);

img_r_half = 888;
img_c_half = 556;

r_offset = -38;
c_offset = 52;

new_img = img(r_cen-img_r_half+r_offset:r_cen+img_r_half+r_offset,c_cen-img_c_half+c_offset:c_cen+img_c_half+c_offset,:);
new_img = imrotate(new_img, 90);
new_img = imresize(new_img, [800, 1280]);

imshow(new_img);

%% Save
f = figure;

roi_x_min = 0;
roi_z_min = -5;
roi_z_max = 3;

list_roi_x = [15, 30, 50, 100, 110];

for i=1:length(list_roi_x)
    roi_x_max = list_roi_x(i);
    roi_y_min = -list_roi_x(i)*0.8;
    roi_y_max = list_roi_x(i)*0.8;

    ptCloud = pointCloud(xyz_1);
    roi = [roi_x_min roi_x_max ...
            roi_y_min roi_y_max ...
            roi_z_min roi_z_max];
    indices = findPointsInROI(ptCloud, roi);
    ptCloud = select(ptCloud, indices);
    
    pcshow(ptCloud, 'BackgroundColor', [1 1 1], 'MarkerSize', 15)
    view(2)
    xlim([roi_x_min, roi_x_max])
    ylim([roi_y_min, roi_y_max])
    
    grid off
    axis off
    
    hgexport(f(1), sprintf('temp'), hgexport('factorystyle'), 'Format', 'png','Resolution',500,'ShowUI','off');
    
    img = imread('temp.png');
    [r, c, channel] = size(img);
    
    r_cen = round(r/2);
    c_cen = round(c/2);
    
    img_r_half = 888;
    img_c_half = 556;
    
    r_offset = -38;
    c_offset = 52;
    
    new_img = img(r_cen-img_r_half+r_offset:r_cen+img_r_half+r_offset,c_cen-img_c_half+c_offset:c_cen+img_c_half+c_offset,:);
    new_img = imrotate(new_img, 90);
    new_img = imresize(new_img, [800, 1280]);

    img_name = strcat(pathSave, 'pcimg_', num2str(list_roi_x(i)), '_', num2str(time), '.png');
    
    imwrite(new_img, img_name)
end


