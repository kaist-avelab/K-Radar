import numpy as np
import torch
from box_intersection_2d import box_intersection_th, box_in_box_th, oriented_box_intersection_2d
from utiles import box2corners, box_intersection, box_in_box, box_intersection_area

def test_box_intersection_th():
    box1 = [0, 0, 2, 3, np.pi/6]
    box2 = [1, 1, 4, 4, -np.pi/4]
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.stack([corners1, corners2], axis=0)) # 2, 4, 2
    tensor1 = torch.unsqueeze(tensor1, 0) # 1, 2, 4, 2
    tensor2 = torch.FloatTensor(np.stack([corners2, corners1], axis=0))
    tensor2 = torch.unsqueeze(tensor2, 0)

    inters, mask = box_intersection_th(tensor1, tensor2)
    inters1 = inters[0,0,...]
    mask1 = mask[0,0,...]

    inters2, mask2 = box_intersection(corners1, corners2)
    error = inters1.numpy() - inters2
    max_error = np.max(error)
    print("error of box_intersection_th:")
    print(error)
    print("max error:", max_error)

def test_box_in_box_th():
    box1 = [0, 0, 2, 3, np.pi/6]
    box2 = [1, 1, 4, 4, -np.pi/4]
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.stack([corners1, corners2], axis=0)) # 2, 4, 2
    tensor1 = torch.unsqueeze(tensor1, 0) # 1, 2, 4, 2
    tensor2 = torch.FloatTensor(np.stack([corners2, corners1], axis=0))
    tensor2 = torch.unsqueeze(tensor2, 0)

    c12_th, c21_th = box_in_box_th(tensor1, tensor2)
    c12_np, c21_np = box_in_box(corners1, corners2)

    error1 = c12_th[0,0, ...].numpy() == c12_np
    error2 = c21_th[0,0, ...].numpy() == c21_np
    print()
    print("Does box_in_box_th give same result as np implementation?")
    print(error1)
    print(error2)

def test_area():
    box1 = [0, 0, 2, 3, np.pi/6]
    box2 = [1, 1, 4, 4, -np.pi/4]
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.stack([corners1, corners1], axis=0)) # 2, 4, 2 # test same box
    tensor1 = torch.unsqueeze(tensor1, 0).repeat([2,1,1,1]).cuda() # 2, 2, 4, 2
    tensor2 = torch.FloatTensor(np.stack([corners2, corners1], axis=0))
    tensor2 = torch.unsqueeze(tensor2, 0).repeat([2,1,1,1]).cuda()

    area, vertices = oriented_box_intersection_2d(tensor1, tensor2)
    area_cpu = area.detach().cpu().numpy()
    vertices = vertices.detach().cpu().numpy()
    print("CUDA: ")
    print(area_cpu)
    print(vertices[0,0,...])
    # NOTE: coordinate of vertices might be different because of the normalization in numpy implementation

    print()
    print("Numpy: ")
    area_np, vertices_np = box_intersection_area(box1, box2)
    print(area_np)
    print(vertices_np) 


if __name__ == "__main__":
    # test_box_intersection_th()
    # test_box_in_box_th()
    test_area()