%% Read point cloud from bag
filePath = strcat(pathBaseDir, nameFolderLidarBag);
fileName = strcat(filePath, cellNameFiles{idxFile}, nameTailLidarBag);
pathRosbag = strcat(fileName);
bag = rosbag(pathRosbag); % Matlab tables

%% Point Cloud
bagPc = select(bag, 'Topic', '/os_cloud_node/points');
listTimestampPc = bagPc.MessageList{:,1};
cellPoints = readMessages(bagPc);
[numFramesPc, tempDim] = size(cellPoints);
listBevRange = [15, 30, 50, 100, 110];
listMarkerSize = [8, 10, 15, 15, 15];
roi_x_min = 0;
roi_z_min = -5;
roi_z_max = 3;

%% Camera Image
bagImg = select(bag, 'Topic', '/usb_cam/image_raw/compressed');
listTimestampImg = bagImg.MessageList{:,1};
cellImgs = readMessages(bagImg);
[numFramesImg, tempDim] = size(cellImgs);

%% Get Point Cloud
strTimestampPc = '';
for idxPc = 1:numFramesPc
    idxPc
    namePcd = strcat('pc_', num2str(idxPc, '%05.f'), '.pcd');
    pathPcd = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\lidar_point_cloud\', namePcd);
    
    tempPc = cellPoints{idxPc};
    tempXyz = readXYZ(tempPc);
    tempIntensity = readField(tempPc, 'intensity');
    
    tempPtCloud = pointCloud(tempXyz, 'Intensity', tempIntensity);
    pcwrite(tempPtCloud, pathPcd)
    
    temp_str_pc = strcat(num2str(idxPc, '%05.f'), ',', namePcd, ',', num2str(listTimestampPc(idxPc), '%.18f'));
    strTimestampPc = strcat(strTimestampPc, temp_str_pc, '\n');

    for idxXrange = 1:length(listBevRange)
        fig1 = figure(1);
        set(fig1, 'Position', [100,100,800,1200])

        roi_x_max = listBevRange(idxXrange);
        roi_y_min = -listBevRange(idxXrange)*0.8;
        roi_y_max = listBevRange(idxXrange)*0.8;
        
        tempPtCloud = pointCloud(tempXyz);
        roi = [ roi_x_min roi_x_max ...
                roi_y_min roi_y_max ...
                roi_z_min roi_z_max ];
        tempIndices = findPointsInROI(tempPtCloud, roi);
        tempPtCloud = select(tempPtCloud, tempIndices);

        pcshow(tempPtCloud, 'BackgroundColor', [1 1 1], 'MarkerSize', 10)
        view(2)
        xlim([roi_x_min, roi_x_max])
        ylim([roi_y_min, roi_y_max])

%         grid off
%         axis off
% 
%         drawnow
%         ax = gca;
%         ax.Units = 'pixels';
%         pos = ax.Position;
% 
%         marg = 0;
%         rect = [-marg, -marg, pos(3)+2*marg, pos(4)+2*marg];
%         F = getframe(gca,rect);
%         ax.Units = 'normalized';
        F = getframe(fig1);
        nameMatFile = strcat('lidar_bev_', num2str(roi_x_max), '_', num2str(idxPc, '%05.f'), '.png');
        pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\lidar_bev_image_origin\', nameMatFile);
        imwrite(F.cdata, pathMatFile)
    end
end
nameTextFile = strcat('timestamp_pc', '.txt');
fileID = fopen(strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\info_frames\', nameTextFile), 'w');
fprintf(fileID, strTimestampPc);

%% Get Front Image
strTimestampImg = '';
for idxImg = 1:numFramesImg
    nameImg = strcat('img_', num2str(idxImg, '%05.f'), '.png');
    pathImg = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\front_image\', nameImg);
    
    tempImg = readImage(cellImgs{idxImg});
    imwrite(tempImg, pathImg)
    
    temp_str_pc = strcat(num2str(idxImg, '%05.f'), ',', nameImg, ',', num2str(listTimestampImg(idxImg), '%.18f'));
    strTimestampImg = strcat(strTimestampImg, temp_str_pc, '\n');
end
nameTextFile = strcat('timestamp_img', '.txt');
fileID = fopen(strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\info_frames\', nameTextFile), 'w');
fprintf(fileID, strTimestampImg);
