import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

IS_UBUNTU = False

EPS = np.finfo(float).eps

DIC_CLS_BGR = {
    'Sedan':[0,0,255],
    'Bus or Truck':[0,50,255],
    'Motorcycle':[0,255,0],
    'Bicycle':[0,200,255],
    'Pedestrian':[255,0,0],
    'Pedestrian Group':[255,0,100],
    'Bicycle Group':[255,100,0],
}

DIC_CLS_RGB = {
    'Sedan': [1, 0, 0],
    'Bus or Truck': [1, 0.2, 0],
    'Motorcycle': [0, 1, 0],
    'Bicycle': [1, 0.8, 0],
    'Pedestrian': [0, 0, 1],
    'Pedestrian Group': [0.4, 0, 1],
    'Bicycle Group': [0, 0.8, 1],
}

if __name__ == '__main__':
    print(BASE_DIR)
    print(EPS)
