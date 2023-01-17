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

%% Get Point Cloud
idxPc = 1;
    
tempPc = cellPoints{idxPc};
tempXyz = readXYZ(tempPc);
tempIntensity = readField(tempPc, 'intensity');

idxXrange = 3;

fig1 = figure(1);
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

%%
% grid off
axis off

%%
set(fig1, 'Position', [100,100,800,1200])

%%
ax = gca;
ax.Units = 'pixels';
pos = ax.Position;

rect = [0, 0, pos(3), pos(4)];
F = getframe(ax,rect);
ax.Units = 'normalized';
%%
fig2 = figure(2);
imshow(F.cdata)

