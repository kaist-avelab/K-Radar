import torch 
from pytorch3d import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Line3D, Line3DCollection

from typing import Dict, List

import os

global_pose = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

linestyle_list = ['solid', 'dotted', 'dashed']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def cal_odom_rel(pre_pose:np.ndarray, nxt_pose:np.ndarray)->np.ndarray:
    '''
    :Inputs:
        pre_pose: 4x4 global transformation matrix
        nxt_pose: 4x4 global transformation matrix
    :Returns:
        odom_rel: 7, local quaternion and position
    '''
    odom_rel = torch.tensor(np.linalg.inv(pre_pose) @ nxt_pose)    
    q = transforms.matrix_to_quaternion(odom_rel[:3,:3])
    t = odom_rel[:3, 3]
    return torch.cat([q, t], dim=0).detach().cpu().numpy()
    
def animation_update(num:int, 
                     global_poses:List[np.ndarray], 
                     lines:List[Line3D], 
                     scatters:List[Path3DCollection], 
                     frames: List[Dict[str, Line3DCollection]]):
    
    

        for idx in range(len(global_poses)):
            
            poses, line, scatter, frame = global_poses[idx], lines[idx][0], scatters[idx], frames[idx]
            
            if num >= poses.shape[0]: continue

            line.set_data(poses[:num+1, 0, 3], poses[:num+1, 1, 3])
            line.set_3d_properties(poses[:num+1, 2, 3])
            
            scatter.set_offsets((poses[num,0,3], poses[num,1,3]))
            scatter.set_3d_properties(poses[num,2,3], 'z')



            new_pose = global_pose @ poses[num]
            local_frame_biases = new_pose[:3,:3] @ np.array([[1., 0., 0.,],
                                                            [0., 1., 0.,],
                                                            [0., 0., 1.,]]) * 100 #* global frame의 bias (e1, e2, e3)를 곱하여 local frame의 bias로 표현

            frame['x_axis'].set_segments([np.array([[poses[num,0,3], poses[num,1,3], poses[num,2,3]], 
                                                    [poses[num,0,3] + local_frame_biases[0,0],
                                                    poses[num,1,3] + local_frame_biases[1,0],
                                                    poses[num,2,3] + local_frame_biases[2,0]]])])
            frame['y_axis'].set_segments([np.array([[poses[num,0,3], poses[num,1,3], poses[num,2,3]], 
                                                    [poses[num,0,3] + local_frame_biases[0,1],
                                                    poses[num,1,3] + local_frame_biases[1,1],
                                                    poses[num,2,3] + local_frame_biases[2,1]]])])
            frame['z_axis'].set_segments([np.array([[poses[num,0,3], poses[num,1,3], poses[num,2,3]], 
                                                    [poses[num,0,3] + local_frame_biases[0,2],
                                                    poses[num,1,3] + local_frame_biases[1,2],
                                                    poses[num,2,3] + local_frame_biases[2,2]]])])
        
        return lines, scatters, frame


def get_global_poses(poses:np.ndarray, mode='local'):
    # quaternion and position [qw, qx, qy, qz, tx, ty, tz]
    if poses.shape[-1] == 7:        
        data = np.zeros((poses.shape[0], 4, 4), dtype=np.float32)
        R = transforms.quaternion_to_matrix(torch.tensor(poses[:, :4])).detach().cpu().numpy()        
        data[:,:3,:3] = R
        data[:,:3, 3] = poses[:, 4:]
        data[:, 3, 3] = np.ones((poses.shape[0]))
        poses = data
    # Otherwise, it would be given 4x4 tranformation matrix
    
    if mode == 'local':
        data = np.zeros((poses.shape[0], 4, 4), dtype=np.float32)
        acc_T = poses[0] #* 가장 처음 rel_pose가 eye matrix인 경우
        data[0] = poses[0]
        
        for idx in range(1, len(poses)):
            acc_T = acc_T @ poses[idx]
            data[idx] = acc_T        
        poses = data        
    # Otherwise, it would be given global coordinates
    
    return poses


def odom_vis_from_multiple_sequences(multiple_poses):
    
    multiple_global_poses = []
    maxlen_pose = 0
    for (poses, mode, linestyle, color) in multiple_poses:
        multiple_global_poses.append([get_global_poses(poses, mode), linestyle, color])
        if maxlen_pose < poses.shape[0]:
            maxlen_pose = poses.shape[0]

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(projection='3d')
    xlim = [-800.0, 800.0]
    ylim = [-800.0, 800.0]
    zlim = [-800.0, 800.0]
    ax.set_xlim3d(xlim)
    ax.set_xlabel('X')
    ax.set_ylim3d(ylim)
    ax.set_ylabel('Y')
    ax.set_zlim3d(zlim)
    ax.set_zlabel('Z')
    global_frame = {'x_axis': ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.1),
                    'y_axis': ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1),
                    'z_axis': ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1)}
    
    lines = []
    scatters = []
    local_frames = []
    global_poses = []
    
    for (poses,linestyle,color) in multiple_global_poses:

        lines.append(ax.plot(poses[0:1, 0, 3], poses[0:1, 1, 3], poses[0:1, 2, 3], linestyle=linestyle, c=color))
        scatters.append(ax.scatter(poses[:1,0,3], poses[:1,1,3], poses[:1,2,3], c='r', s=50)) # 빨간색 점
        local_frames.append({'x_axis': ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.1),
                            'y_axis': ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1),
                            'z_axis': ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1)})
        global_poses.append(poses)

        arrow = ax.quiver(0,0,0, 0,0,0, color='r', arrow_length_ratio=0.1)
        
    ani = animation.FuncAnimation(fig, animation_update, maxlen_pose, fargs=(global_poses, lines, scatters, local_frames), interval=10, blit=False, repeat=False)
    plt.show()

def odom_vis_from_filepath(odom_path):
    
    global_pose = np.array([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
    
    with open(odom_path, 'r') as f:
        lines = f.readlines()
        poses = np.zeros((len(lines), 4, 4))

        for idx, line in enumerate(lines):
            poses[idx] = np.array([float(element) for element in line.rstrip().split(' ')] + [0., 0., 0., 1.]).reshape(4, 4)
    print(f'{odom_path} sequence length: {poses.shape[0]}')
                        
    def update(num:int, poses:np.ndarray, line:Line3D, scatter:Path3DCollection, frame: Dict[str, Line3DCollection]):

        line.set_data(poses[:num+1, 0, 3], poses[:num+1, 1, 3])
        line.set_3d_properties(poses[:num+1, 2, 3])
        
        scatter.set_offsets((poses[num,0,3], poses[num,1,3]))
        scatter.set_3d_properties(poses[num,2,3], 'z')



        new_pose = global_pose @ poses[num]
        local_frame_biases = new_pose[:3,:3] @ np.array([[1., 0., 0.,],
                                                         [0., 1., 0.,],
                                                         [0., 0., 1.,]]) * 100 #* global frame의 bias (e1, e2, e3)를 곱하여 local frame의 bias로 표현

        frame['x_axis'].set_segments([np.array([[poses[num,0,3], poses[num,1,3], poses[num,2,3]], 
                                                [poses[num,0,3] + local_frame_biases[0,0],
                                                 poses[num,1,3] + local_frame_biases[1,0],
                                                 poses[num,2,3] + local_frame_biases[2,0]]])])
        frame['y_axis'].set_segments([np.array([[poses[num,0,3], poses[num,1,3], poses[num,2,3]], 
                                                [poses[num,0,3] + local_frame_biases[0,1],
                                                 poses[num,1,3] + local_frame_biases[1,1],
                                                 poses[num,2,3] + local_frame_biases[2,1]]])])
        frame['z_axis'].set_segments([np.array([[poses[num,0,3], poses[num,1,3], poses[num,2,3]], 
                                                [poses[num,0,3] + local_frame_biases[0,2],
                                                 poses[num,1,3] + local_frame_biases[1,2],
                                                 poses[num,2,3] + local_frame_biases[2,2]]])])
        
        return line, scatter, frame
        
    fig = plt.figure(figsize=(24,16))
    ax = fig.add_subplot(projection='3d')
    line, = ax.plot(poses[0:1, 0, 3], poses[0:1, 1, 3], poses[0:1, 2, 3])
    scatter = ax.scatter(poses[:1,0,3], poses[:1,1,3], poses[:1,2,3], c='r', s=50)  # 빨간색 점
    
    global_frame = {'x_axis': ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.1),
                    'y_axis': ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1),
                    'z_axis': ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1)}
    
    local_frame = {'x_axis': ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.1),
                   'y_axis': ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1),
                   'z_axis': ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1)}
    
    arrow = ax.quiver(0,0,0, 0,0,0, color='r', arrow_length_ratio=0.1)

    xlim = [-800.0, 800.0]
    ylim = [-800.0, 800.0]
    zlim = [-800.0, 800.0]
    ax.set_xlim3d(xlim)
    ax.set_xlabel('X(m)')
    ax.set_ylim3d(ylim)
    ax.set_ylabel('Y(m)')
    ax.set_zlim3d(zlim)
    ax.set_zlabel('Z(m)')
        
    ani = animation.FuncAnimation(fig, update, len(lines), fargs=(poses, line, scatter, local_frame), interval=10, blit=False, repeat=False)
    plt.show()


if __name__ == '__main__':
    
    for i in range(1, 59):
        odom_path = f'../gt_pose/gt_{str(i).zfill(2)}.txt'
        odom_vis_from_filepath(odom_path)      


            
    