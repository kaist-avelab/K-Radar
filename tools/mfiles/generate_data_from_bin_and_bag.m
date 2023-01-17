%% Load files
clear, clc, close all
pathBaseDir = 'E:\radar_bin_lidar_bag_files\';
cellNameFiles = ...
    {'static_1', ...
     'static_2', ...
     'static_3', ...
     };

nameFolderRadarBin = 'radar_bin_files\';
nameTailRadarBin = '.bin';
nameFolderLidarBag = 'lidar_bag_files\';
nameTailLidarBag = '.bag';

load('info_arr.mat') % arrRange, arrAzimuth, arrElevation
idxFile = 1; % 'static_1'

%% Processing radar tessurect
% arrDREA (4D Tesseract)
restoredefaultpath
addpath(genpath(pwd))

setUserDefinedParams;
setRadarParams;
srsAmanda(hwCfg, radarCfg, aoaCfg, 'init');

filePath = strcat(pathBaseDir, nameFolderRadarBin);
fileName = strcat(filePath, cellNameFiles{idxFile}, '_');

%% numFrames: Frame 개수
minFileSize = Inf;
for chipIdx = 1:hwCfg.numChips
    currentFileName = strcat(fileName, num2str(chipIdx), nameTailRadarBin);
    currentFileInfo = dir(currentFileName);
    currentFileSize = currentFileInfo.bytes;
    if (currentFileSize < minFileSize)
        minFileSize = currentFileSize;
    end
end
numFrames = floor(minFileSize/frameSizeInByte);
radarCube = single(zeros(radarCfg.numAdcSamples, hwCfg.numRxAntsPerChip, radarCfg.numChirpsPerLoop, radarCfg.numChirpLoops, hwCfg.numChips));

%% Iterate per frames
fprintf('Total frames = %d ...\n', numFrames)
cellPathRadarFiles = {};
for frameIdx = 1:numFrames
    fprintf('frameIdx = %d is being processed ...\n', frameIdx)
    for chipIdx = 1:hwCfg.numChips
        adcData = readAdcData(frameSizeInByte, fileName, chipIdx, frameIdx);             
        radarCube(:, :, :, :, chipIdx) = reshape(adcData(1:2:end), radarCfg.numAdcSamples, ...
                                                                   hwCfg.numRxAntsPerChip, ...
                                                                   radarCfg.numChirpsPerLoop, ...
                                                                   radarCfg.numChirpLoops) ...
                                       + 1j* ...
                                         reshape(adcData(2:2:end), radarCfg.numAdcSamples, ...
                                                                   hwCfg.numRxAntsPerChip, ...
                                                                   radarCfg.numChirpsPerLoop, ...
                                                                   radarCfg.numChirpLoops); 
    end
    radarCube = srsAmanda(hwCfg, radarCfg, aoaCfg, 'calib', radarCube); % calibration
    rangeFFTout = fft(radarCube, dspCfg.numRangeBins, 1); % range FFT (w/o windowing)
    dopplerFFTout = fft(rangeFFTout, dspCfg.numDopplerBins, 4); % doppler FFT (w/o windowing)
    sizeRadarCube = size(dopplerFFTout); % Doppler-Range-Elevation-Azimuth Tensor (Tessurect)
    binRange = sizeRadarCube(1);
    binDoppler = sizeRadarCube(4);
    
    arrDREA = zeros(binDoppler,binRange,37,107,'single');
    for idxDoppler = 1:binDoppler
        for idxRange = 1:binRange
            aoaInput = squeeze(dopplerFFTout(idxRange, :, :, idxDoppler, :));
            arrDREA(idxDoppler,idxRange,:,:) = srsAmanda(hwCfg, radarCfg, aoaCfg, 'dbf', aoaInput, idxDoppler-1, dspCfg);
        end
    end
    % arrDREA = abs(arrDREA);
    arrDREA = double(arrDREA);
    
    % flip along elevation
    arrDREA = flip(arrDREA,3);

    nameMatFile = strcat(cellNameFiles{idxFile}, '_', num2str(frameIdx, '%05.f'), '.mat');
    pathMatFile = strcat(pathBaseDir, 'generated_files\radar_tesseract\', nameMatFile);
    cellPathRadarFiles{end+1} = pathMatFile;
    save(pathMatFile, 'arrDREA')
end
nameMatFile = strcat(cellNameFiles{idxFile}, '_', 'cell_DREA', '.mat');
pathMatFile = strcat(pathBaseDir, 'generated_files\cell_path\', nameMatFile);
save(pathMatFile, 'cellPathRadarFiles')

%% Processing REA -> BEV
% load(pathMatFile)
load('E:\radar_bin_lidar_bag_files\generated_files\cell_path\static_1_cell_DREA.mat')
cellPathRadarZyxCube = {};

%%% Hyperparams %%%
x_min = 0; % [m]
x_per_bin = 0.2; % [m]
x_max = 100-x_per_bin; % [m]

y_min = -80; % [m]; -x_max*sin(53*pi/180)
y_per_bin = 0.2; % [m]
y_max = 80-y_per_bin; % [m]; x_max*sin(53*pi/180)

z_min = -30; % [m]; -x_max*sin(18*pi/180) = -30.9
z_per_bin = 0.2; % [m]
z_max = 30-z_per_bin; % [m]; x_max*sin*18*pi/180) = 30.9

arr_x = x_min:x_per_bin:x_max;
arr_y = y_min:y_per_bin:y_max;
arr_z = z_min:z_per_bin:z_max;

len_x = length(arr_x);
len_y = length(arr_y);
len_z = length(arr_z);

RAD2DEG = 180/pi;
DEG2RAD = pi/180;
%%% Hyperparams %%%

fprintf('Total frames = %d ...\n', length(cellPathRadarFiles))
for idxPath = 1:1%length(cellPathRadarFiles)
    fprintf('pathIdx = %d is being processed ...\n', idxPath)
    arrDreaStruct = load(cellPathRadarFiles{idxPath});
    arrDreaTemp = arrDreaStruct.arrDREA;

    %%% Mean doppler around zero doppler %%%
    % Tessurect -> 3D Tensor (DREA -> REA)
    % idxDopplerOffset = 10;
    % arrDreaTemp = arrDreaTemp(1:idxDopplerOffset,:,:,:);
    %%% Mean doppler around zero doppler %%%

    arrREA = squeeze(mean(arrDreaTemp,1));

    %%% REA -> BEV %%%
    arr_zyx = -1*ones(len_z,len_y,len_x);
    
    r_max = max(arrRange);
    r_min = min(arrRange);
    a_max = max(arrAzimuth)*DEG2RAD; % [rad]
    a_min = min(arrAzimuth)*DEG2RAD; % [rad]
    e_max = max(arrElevation)*DEG2RAD; % [rad]
    e_min = min(arrElevation)*DEG2RAD; % [rad]
    
    arrAzimuth = arrAzimuth.*DEG2RAD;
    arrElevation = arrElevation.*DEG2RAD;
    
    cnt1 = 0;
    cnt2 = 0;
    for i_z = 1:len_z
        for i_y = 1:len_y
            for i_x = 1:len_x
                z = arr_z(i_z);
                y = arr_y(i_y);
                x = arr_x(i_x);
                
                [r,a,e] = c2p(x,y,z); % [rad]
    
                % exception 1
                if (r<r_min) || (r>r_max) || ...
                    (a<a_min) || (a>a_max) || ...
                    (e<e_min) || (e>e_max)
                    %fprintf('exception! cnt = %d\n',cnt)
                    cnt1 = cnt1 + 1;
                    continue
                end
    
                % find index (integer) of r, a, e
                [i_r0, i_r1] = findIndexForBiInt(r, arrRange);
                [i_a0, i_a1] = findIndexForBiInt(a, arrAzimuth);
                [i_e0, i_e1] = findIndexForBiInt(e, arrElevation);
    
                % exception 2
                if i_r0 == -1 || i_r1 == -1 || ...
                    i_a0 == -1 || i_a1 == -1 || ...
                    i_e0 == -1 || i_e1 == -1
                    cnt2 = cnt2 + 1;
                    continue
                end
    
                % Bilinear Interpolation
                val_rea = [r,e,a];
                arr_bi_index = [[i_r0, i_r1];[i_e0, i_e1];[i_a0, i_a1]];
                val = getBiIntValue(arrREA, val_rea, arr_bi_index, arrRange, arrElevation, arrAzimuth);
                
                % Without Bin Int
                % arr_zyx(i_z,i_y,i_x) = arrREA(i_r0,i_e0,i_a0);
    
                % With Bin Int
                arr_zyx(i_z,i_y,i_x) = val;
            end
        end
    end
    %%% REA -> BEV %%%

    nameMatFile = strcat(cellNameFiles{idxFile}, '_', num2str(idxPath, '%05.f'), '.mat');
    pathMatFile = strcat(pathBaseDir, 'generated_files\radar_zyx_cube\', nameMatFile);
    cellPathRadarZyxCube{end+1} = pathMatFile;
    save(pathMatFile, 'arr_zyx')
end
nameMatFile = strcat(cellNameFiles{idxFile}, '_', 'cell_ZYX', '.mat');
pathMatFile = strcat(pathBaseDir, 'generated_files\cell_path\', nameMatFile);
save(pathMatFile, 'cellPathRadarZyxCube')

%% Get the BEV image (Mean along Z-axis) 
[len_z, len_y, len_x] = size(arr_zyx);

cnt_minus_1 = 0;
cnt_minus = 0;

new_arr_xy = zeros(len_y, len_x);

for i_y = 1:len_y
    for i_x = 1:len_x
        temp_arr_z = squeeze(arr_zyx(:,i_y,i_x));
        pw_sum = 0;
        cnt_none_minus_1 = 0;
        for i_z = 1:len_z
            if temp_arr_z(i_z) < 0
                cnt_minus = cnt_minus + 1;
                continue
            end
            if temp_arr_z(i_z) == -1
                cnt_minus_1 = cnt_minus_1 + 1;
                continue
            end
            pw_sum = pw_sum + temp_arr_z(i_z);
            cnt_none_minus_1 = cnt_none_minus_1 + 1;
        end
        new_arr_xy(i_y,i_x) = pw_sum/cnt_none_minus_1;
    end
end

%% Set ROI & Visualization (XY)
roi_x_min = 0;
roi_x_max = 100;
roi_y_min = -80;
roi_y_max = 80;

% Radar
[i_x_min, i_x_max, x_new] = findRoiIndex(roi_x_min,roi_x_max,arr_x);
[i_y_min, i_y_max, y_new] = findRoiIndex(roi_y_min,roi_y_max,arr_y);
arr_xy_vis = new_arr_xy(i_y_min:i_y_max,i_x_min:i_x_max);

% XY
fig1 = figure;
surf(x_new, y_new, 10*log10(arr_xy_vis));
% surf(arr_x, arr_y, 10*log10(new_arr_xy));
shading interp
colormap jet
view(2)
xlim([roi_x_min, roi_x_max])
ylim([roi_y_min, roi_y_max])
% axis off
grid off

% save('temp_plot.mat', 'x_new', 'y_new', 'arr_xy_vis', 'roi_x_min', 'roi_x_max', 'roi_y_min', 'roi_y_max')

%% Get the BEV image (Mean along Y-axis for partial tensor)
[len_z, len_y, len_x] = size(arr_zyx);

roi_x_min = 0;
roi_x_max = 50;
roi_y_min = -4;
roi_y_max = -3;
roi_z_min = -15;
roi_z_max = 15;

% Radar
[i_x_min, i_x_max, x_new] = findRoiIndex(roi_x_min,roi_x_max,arr_x);
[i_y_min, i_y_max, y_new] = findRoiIndex(roi_y_min,roi_y_max,arr_y);
[i_z_min, i_z_max, z_new] = findRoiIndex(roi_z_min,roi_z_max,arr_z);

new_arr_vis = arr_zyx(i_z_min:i_z_max,i_y_min:i_y_max,i_x_min:i_x_max);
[new_len_z, new_len_y, new_len_x] = size(new_arr_vis);

new_arr_xz = zeros(new_len_z, new_len_x);

for i_z = 1:new_len_z
    for i_x = 1:new_len_x
        temp_arr_y = squeeze(new_arr_vis(i_z,:,i_x));
        pw_sum = 0;
        cnt_none_minus_1 = 0;
        for i_y = 1:new_len_y
            if temp_arr_y(i_y) < 0
                continue
            end
            pw_sum = pw_sum + temp_arr_y(i_y);
            cnt_none_minus_1 = cnt_none_minus_1 + 1;
        end
        new_arr_xz(i_z,i_x) = pw_sum/cnt_none_minus_1;
    end
end

% XZ
fig1 = figure;
surf(x_new, z_new, 10*log10(new_arr_xz));
shading interp
colormap jet
view(2)
xlim([roi_x_min, roi_x_max])
ylim([roi_z_min, roi_z_max])
grid off

%% Function
% Cart to polar
function [r,a,e] = c2p(x,y,z)
    % in:
    %   x, y, z [m]
    % out:
    %   range [m], azimuth, elevation [rad]
    r = sqrt(x^2+y^2+z^2);
    a = atan(-y/x);
    e = atan(z/sqrt(x^2+y^2));
end

function [x,y,z] = p2c(r,a,e)
    % in, out: opposite of c2p
    x = r*cos(a)*cos(e);
    y = r*sin(a)*cos(e);
    z = r*sin(e);
end

% Find index for bilinear interpolation
function [i_0, i_1] = findIndexForBiInt(value, array)
    i_0 = -1;
    i_1 = -1;
    for i = 1:(length(array)-1)
        if ((value > array(i)) && (value <= array(i+1)))
            i_0 = i;
            i_1 = i+1;
            break
        end
    end
end

function v = getBiIntValue(rea, val_rea, arr_bi_index, arrRange, arrElevation, arrAzimuth)
    r = val_rea(1);
    e = val_rea(2);
    a = val_rea(3);

    i_r = arr_bi_index(1,:);
    i_e = arr_bi_index(2,:);
    i_a = arr_bi_index(3,:);
    
    i_r0 = i_r(1);
    i_r1 = i_r(2);
    i_e0 = i_e(1);
    i_e1 = i_e(2);
    i_a0 = i_a(1);
    i_a1 = i_a(2);

    rea000 = rea(i_r0,i_e0,i_a0);
    rea001 = rea(i_r0,i_e0,i_a1);
    rea010 = rea(i_r0,i_e1,i_a0);
    rea011 = rea(i_r0,i_e1,i_a1);
    rea100 = rea(i_r1,i_e0,i_a0);
    rea101 = rea(i_r1,i_e0,i_a1);
    rea110 = rea(i_r1,i_e1,i_a0);
    rea111 = rea(i_r1,i_e1,i_a1);
    
    del_r = arrRange(i_r1) - arrRange(i_r0);
    del_e = arrElevation(i_e1) - arrElevation(i_e0);
    del_a = arrAzimuth(i_a1) - arrAzimuth(i_a0);

    v = 1/(del_r*del_e*del_a)*( ...
        rea000*((arrRange(i_r1)-r)*(arrElevation(i_e1)-e)*(arrAzimuth(i_a1)-a))+ ...
        rea001*((arrRange(i_r1)-r)*(arrElevation(i_e1)-e)*(a-arrAzimuth(i_a0)))+ ...
        rea010*((arrRange(i_r1)-r)*(e-arrElevation(i_e0))*(arrAzimuth(i_a1)-a))+ ...
        rea011*((arrRange(i_r1)-r)*(e-arrElevation(i_e0))*(a-arrAzimuth(i_a0)))+ ...
        rea100*((r-arrRange(i_r0))*(arrElevation(i_e1)-e)*(arrAzimuth(i_a1)-a))+ ...
        rea101*((r-arrRange(i_r0))*(arrElevation(i_e1)-e)*(a-arrAzimuth(i_a0)))+ ...
        rea110*((r-arrRange(i_r0))*(e-arrElevation(i_e0))*(arrAzimuth(i_a1)-a))+ ...
        rea111*((r-arrRange(i_r0))*(e-arrElevation(i_e0))*(a-arrAzimuth(i_a0))) ...
        );
    
end

function [i_min, i_max, new_arr] = findRoiIndex(v_min, v_max, arr)
    i_min = find(arr==v_min);
    i_max = find(arr==v_max);
    new_arr = arr(i_min:i_max);
end
