import numpy as np
from scipy.io import loadmat # from matlab

if __name__ == '__main__':
    ### Settings ###
    temp_values = loadmat('./resources/info_arr.mat')
    arr_range = temp_values['arrRange']
    deg2rad = np.pi/180.
    arr_azimuth = temp_values['arrAzimuth']*deg2rad
    arr_elevation = temp_values['arrElevation']*deg2rad
    _, num_0 = arr_range.shape
    _, num_1 = arr_azimuth.shape
    _, num_2 = arr_elevation.shape
    arr_range = arr_range.reshape((num_0,))
    arr_azimuth = arr_azimuth.reshape((num_1,))
    arr_elevation = arr_elevation.reshape((num_2,))

    n_r = len(arr_range)
    n_a = len(arr_azimuth)
    n_e = len(arr_elevation)
    rae_r = np.repeat(np.repeat((arr_range).copy().reshape(n_r,1,1),n_a,1),n_e,2)
    rae_a = np.repeat(np.repeat((arr_azimuth).copy().reshape(1,n_a,1),n_r,0),n_e,2)
    rae_e = np.repeat(np.repeat((arr_elevation).copy().reshape(1,1,n_e),n_r,0),n_a,1)

    # Radar polar to General polar coordinate
    rae_a = -rae_a
    rae_e = -rae_e
    
    # For flipped azimuth & elevation angle
    xyz_x = rae_r * np.cos(rae_e) * np.cos(rae_a)
    xyz_y = rae_r * np.cos(rae_e) * np.sin(rae_a)
    xyz_z = rae_r * np.sin(rae_e)

    rdr_polar_3d_xyz = np.stack((xyz_x,xyz_y,xyz_z),axis=0)
    ### Settings ###

    ### Load 
    polar3d = np.load('./tools/polar3d_checker/polar3d_00033.npy')
    print(polar3d.shape)