clc, clear, close all

%%% Step 1: Default & Hyper-parameters %%%
gen_1_load_data

idxFile = 1;

gen_5_get_pc_img

clc, clear, close all

gen_1_load_data

idxFile = 2;

gen_5_get_pc_img

%%
%%% Step 2: Tesseract Generation %%%
gen_2_get_tesseract

%%% Step 3: ZYX Cube Generation %%%
% load(strcat(pathBaseDir,'generated_files\cell_path\',...
%                     cellNameFiles{idxFile},'_cell_DREA.mat'))
gen_3_get_zyx_cube

%%% Step 4: BEV Img Generation %%%
% load(strcat(pathBaseDir,'generated_files\',cellNameFiles{idxFile},'\cell_path\',cellNameFiles{idxFile},'_cell_ZYX.mat'))
gen_4_get_bev_img

%%
%%% Step 5: Point Cloud Img Generation %%%
gen_5_get_pc_img
