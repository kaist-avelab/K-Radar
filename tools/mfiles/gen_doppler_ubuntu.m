clc, clear, close all

%%% Dummy value is -10 !!! %%%

genDopplerDir = '/media/oem/HDD_3_2/Radar_Data_Doppler';
cellPathBaseDir = {'/media/oem/data_21/radar_bin_lidar_bag_files/generated_files'};

warning('off', 'MATLAB:MKDIR:DirectoryExists');
mkdir(genDopplerDir);

%%% Hyperparams %%%
load('info_arr.mat') % arrRange, arrAzimuth, arrElevation
load('arr_doppler.mat')
RAD2DEG = 180/pi;
DEG2RAD = pi/180;
arrAzimuthRad = arrAzimuth.*DEG2RAD;
arrElevationRad = arrElevation.*DEG2RAD;

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

% fprintf('Total frames = %d ...\n', numFrames)

for idx_dir = 1:length(cellPathBaseDir)
    pathBaseDir = cellPathBaseDir{idx_dir};
    cellSeqNames = dir(pathBaseDir);
    cellSeqNames = cellSeqNames(3:end);

    for idx_seq = 1:length(cellSeqNames)
        st_seq = cellSeqNames(idx_seq);
        name_seq = st_seq.name;
        path_tesseract = [st_seq.folder, '/', st_seq.name, '/radar_tesseract'];
        list_rdrtes = dir(path_tesseract);
        list_rdrtes = list_rdrtes(3:end);
        total_rdr = length(list_rdrtes);
        dir_gen_dop = [genDopplerDir, '/', name_seq, '/', 'radar_cube_doppler'];
        mkdir(dir_gen_dop)

        for idx_rdr = 1:total_rdr
            st_rdr = list_rdrtes(idx_rdr);
            path_rdr = [st_rdr.folder, '/', st_rdr.name];
            file_idx = split(st_rdr.name, '_');
            file_idx = split(file_idx(2), '.');
            file_idx = file_idx{1};
            fprintf('* seq: %d, rdr: %d/%d ...\n',idx_seq,idx_rdr,total_rdr)

            %%% proces rdr cube to doppler %%%
            load(path_rdr)
            [len_d, len_r, len_e, len_a] = size(arrDREA);
            arrREA = zeros([len_r, len_e, len_a]);
            for idx_r = 1:len_r
                for idx_e = 1: len_e
                    for idx_a = 1:len_a
                        arr_dop_temp = arrDREA(:,idx_r,idx_e,idx_a);
                        [~, max_idx] = max(arr_dop_temp);
                        arrREA(idx_r,idx_e,idx_a) = arr_doppler(max_idx);
                    end
                end
            end

            %%% REA -> BEV %%%
            %%% dummy value as -10 in doppler
            arr_zyx = -10*ones(len_z,len_y,len_x);

            r_max = max(arrRange);
            r_min = min(arrRange);
            a_max = max(arrAzimuth)*DEG2RAD; % [rad]
            a_min = min(arrAzimuth)*DEG2RAD; % [rad]
            e_max = max(arrElevation)*DEG2RAD; % [rad]
            e_min = min(arrElevation)*DEG2RAD; % [rad]

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
                        [i_a0, i_a1] = findIndexForBiInt(a, arrAzimuthRad);
                        [i_e0, i_e1] = findIndexForBiInt(e, arrElevationRad);

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
                        val = getBiIntValue(arrREA, val_rea, arr_bi_index, arrRange, arrElevationRad, arrAzimuthRad);

                        % Without Bin Int
                        % arr_zyx(i_z,i_y,i_x) = arrREA(i_r0,i_e0,i_a0);

                        % With Bin Int
                        arr_zyx(i_z,i_y,i_x) = val;
                    end
                end
            end
            %%% proces rdr cube to doppler %%%

            path_gen_dop = [dir_gen_dop, '/', 'radar_cube_doppler_', file_idx, '.mat'];
            save(path_gen_dop, 'arr_zyx')
        end
    end
end

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
