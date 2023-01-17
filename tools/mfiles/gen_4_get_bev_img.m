%%% Get the BEV image (Mean along Z-axis)
for idxBevImg = 1:length(cellPathRadarZyxCube)
    arr_zyx_struct = load(cellPathRadarZyxCube{idxBevImg});
    arr_zyx = arr_zyx_struct.arr_zyx;

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

    %%% Full Range %%%
    fig1 = figure(1);
    set(fig1, 'Position', [100,100,800,1200])
    surf(arr_x, arr_y, 10*log10(new_arr_xy));
    shading interp
    colormap jet
    view(2)
    xlim([0, 100])
    ylim([-80, 80])
%     axis off
%     grid off

%     drawnow
%     ax = gca;
%     ax.Units = 'pixels';
%     pos = ax.Position;
%     
%     marg = 0;
%     rect = [-marg, -marg, pos(3)+2*marg, pos(4)+2*marg];
%     F = getframe(gca,rect);
%     ax.Units = 'normalized';
    F = getframe(fig1);
    nameMatFile = strcat('radar_bev_100_', num2str(idxBevImg, '%05.f'), '.png');
    pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\radar_bev_image_origin\', nameMatFile);
    imwrite(F.cdata, pathMatFile)

    arr_range = [15, 30, 50];

    %%% Set Range %%%
    for i = 1:length(arr_range)
        roi_x_min = 0;
        roi_x_max = arr_range(i);
        roi_y_min = -int64(arr_range(i)*0.8);
        roi_y_max = int64(arr_range(i)*0.8);
        
        [i_x_min, i_x_max, x_new] = findRoiIndex(roi_x_min,roi_x_max,arr_x);
        [i_y_min, i_y_max, y_new] = findRoiIndex(roi_y_min,roi_y_max,arr_y);
        arr_xy_vis = new_arr_xy(i_y_min:i_y_max,i_x_min:i_x_max);
        
        fig = figure(1);
        set(fig, 'Position', [100,100,800,1200])
        surf(x_new, y_new, 10*log10(arr_xy_vis));
        shading interp
        colormap jet
        view(2)
        xlim([roi_x_min, roi_x_max])
        ylim([roi_y_min, roi_y_max])
%         axis off
%         grid off
    
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
        nameMatFile = strcat('radar_bev_', ...
            num2str(arr_range(i)), '_', num2str(idxBevImg, '%05.f'), '.png');
        pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\radar_bev_image_origin\', nameMatFile);
        imwrite(F.cdata, pathMatFile)
    end
end

%% Function
function [i_min, i_max, new_arr] = findRoiIndex(v_min, v_max, arr)
    [c_min, i_min] = min((double(arr)-double(v_min)).^2);
    [c_max, i_max] = min((double(arr)-double(v_max)).^2);
    new_arr = arr(i_min:i_max);
end
