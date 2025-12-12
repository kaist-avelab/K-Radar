import cv2
import os.path as osp
import os
import numpy as np

# PATH_IMGS = '/media/donghee/HDD_0/K-Radar_fusion_DH/png_ASF_vis/10'

PATH_VIDEO = './video_cam.mp4'
FPS = 20

PATH_IMGS = '/media/donghee/HDD_0/K-Radar_fusion_DH/png_ASF_vis/53'
start_idx = 5
end_idx = 125

if __name__ == '__main__':
    idx = f'{start_idx}'.zfill(5)
    img_test = osp.join(PATH_IMGS, 'CAM', f'{idx}.png')

    img = cv2.imread(img_test)
    img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)

    print(img.shape)

    WIDTH_GAP=320
    HEIGHT_GAP=120
    img = img[HEIGHT_GAP:-HEIGHT_GAP,WIDTH_GAP:-WIDTH_GAP,:]
    
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    size = img.shape[:2]
    fps = 20
    out = cv2.VideoWriter(PATH_VIDEO, fourcc, fps, (size[1], size[0]))
    
    for img_idx in range(start_idx, end_idx):
        print(f"{img_idx}")
        image_file = osp.join(PATH_IMGS, 'CAM', )
        img = cv2.imread(image_file)

        idx = f'{img_idx}'.zfill(5)
        img_test = osp.join(PATH_IMGS, 'CAM', f'{idx}.png')

        img = cv2.imread(img_test)
        img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)

        # print(img.shape)

        # WIDTH_GAP=320
        # HEIGHT_GAP=120
        img = img[HEIGHT_GAP:-HEIGHT_GAP,WIDTH_GAP:-WIDTH_GAP,:]
        
        if img is None:
            print(f"* Can not read the file: {image_file}")
            continue
        
        # if img.shape[1] != size[0] or img.shape[0] != size[1]:
        #     img = cv2.resize(img, size)
        
        out.write(img)
    
    out.release()