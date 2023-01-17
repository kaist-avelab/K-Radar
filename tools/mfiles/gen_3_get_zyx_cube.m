%% Processing REA -> BEV
% load(pathMatFile)
cellPathRadarZyxCube = {};

fprintf('Total frames = %d ...\n', length(cellPathRadarFiles))
for idxPath = 1:length(cellPathRadarFiles)
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
    %%% REA -> BEV %%%

    nameMatFile = strcat('cube_', num2str(idxPath, '%05.f'), '.mat');
    pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\radar_zyx_cube\', nameMatFile);
    cellPathRadarZyxCube{end+1} = pathMatFile;
    save(pathMatFile, 'arr_zyx')
end

%%
nameMatFile = strcat(cellNameFiles{idxFile}, '_', 'cell_ZYX', '.mat');
pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\cell_path\', nameMatFile);
save(pathMatFile, 'cellPathRadarZyxCube')

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
