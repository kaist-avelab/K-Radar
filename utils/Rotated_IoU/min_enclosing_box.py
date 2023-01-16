'''
find the smallest bounding box which enclosing two rectangles. It can be used to calculate the GIoU or DIoU
loss for rotated object detection.

Observation: a side of a minimum-area enclosing box must be collinear with a side of the convex polygon.
https://en.wikipedia.org/wiki/Minimum_bounding_box_algorithms

Since two rectangles have 8 points, brutal force method should be enough. That is, calculate the enclosing box
area for every possible side of the polygon and take the mininum. Their should be 8x7/2 = 28 combinations and 4
of them are impossible (4 diagonal of two boxes). So the function brutally searches in the 24 candidates.

The index of box corners follows the following convention:

  0---1        4---5
  |   |        |   |
  3---2        7---6

author: Lanxiao Li
2020.08
'''

import numpy as np
import torch

def generate_table():
    """generate candidates of hull polygon edges and the the other 6 points

    Returns:
        lines: (24, 2)
        points: (24, 6)
    """
    skip = [[0,2], [1,3], [5,7], [4,6]]     # impossible hull edge
    line = []
    points = []

    def all_except_two(o1, o2):
        a = []
        for i in range(8):
            if i != o1 and i != o2:
                a.append(i)
        return a

    for i in range(8):
        for j in range(i+1, 8):
            if [i, j] not in skip:
                line.append([i, j])
                points.append(all_except_two(i, j))
    return line, points

LINES, POINTS = generate_table()
LINES = np.array(LINES).astype(np.int)
POINTS = np.array(POINTS).astype(np.int)

def gather_lines_points(corners:torch.Tensor):
    """get hull edge candidates and the rest points using the index

    Args:
        corners (torch.Tensor): (..., 8, 2)
    
    Return: 
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)
        idx_lines (torch.Tensor): Long (..., 24, 2, 2)
        idx_points (torch.Tensor): Long (..., 24, 6, 2)
    """
    dim = corners.dim()
    idx_lines = torch.LongTensor(LINES).to(corners.device).unsqueeze(-1)      # (24, 2, 1)
    idx_points = torch.LongTensor(POINTS).to(corners.device).unsqueeze(-1)    # (24, 6, 1)
    idx_lines = idx_lines.repeat(1,1,2)                                       # (24, 2, 2)
    idx_points = idx_points.repeat(1,1,2)                                     # (24, 6, 2)
    if dim > 2:
        for _ in range(dim-2):
            idx_lines = torch.unsqueeze(idx_lines, 0)
            idx_points = torch.unsqueeze(idx_points, 0)
        idx_points = idx_points.repeat(*corners.size()[:-2], 1, 1, 1)           # (..., 24, 2, 2)
        idx_lines = idx_lines.repeat(*corners.size()[:-2], 1, 1, 1)             # (..., 24, 6, 2)
    corners_ext = corners.unsqueeze(-3).repeat( *([1]*(dim-2)), 24, 1, 1)       # (..., 24, 8, 2)
    lines = torch.gather(corners_ext, dim=-2, index=idx_lines)                  # (..., 24, 2, 2)
    points = torch.gather(corners_ext, dim=-2, index=idx_points)                # (..., 24, 6, 2)

    return lines, points, idx_lines, idx_points

def point_line_distance_range(lines:torch.Tensor, points:torch.Tensor):
    """calculate the maximal distance between the points in the direction perpendicular to the line
    methode: point-line-distance

    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)
    
    Return:
        torch.Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]       # (..., 24, 1)
    y1 = lines[..., 0:1, 1]       # (..., 24, 1)
    x2 = lines[..., 1:2, 0]       # (..., 24, 1)
    y2 = lines[..., 1:2, 1]       # (..., 24, 1)
    x = points[..., 0]            # (..., 24, 6)
    y = points[..., 1]            # (..., 24, 6)
    den = (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1
    # NOTE: the backward pass of torch.sqrt(x) generates NaN if x==0
    num = torch.sqrt( (y2-y1).square() + (x2-x1).square() + 1e-14 )
    d = den/num         # (..., 24, 6)
    d_max = d.max(dim=-1)[0]       # (..., 24)
    d_min = d.min(dim=-1)[0]       # (..., 24)
    d1 = d_max - d_min             # suppose points on different side
    d2 = torch.max(d.abs(), dim=-1)[0]      # or, all points are on the same side
    # NOTE: if x1 = x2 and y1 = y2, this will return 0
    return torch.max(d1, d2)

def point_line_projection_range(lines:torch.Tensor, points:torch.Tensor):
    """calculate the maximal distance between the points in the direction parallel to the line
    methode: point-line projection

    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)
    
    Return:
        torch.Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]       # (..., 24, 1)
    y1 = lines[..., 0:1, 1]       # (..., 24, 1)
    x2 = lines[..., 1:2, 0]       # (..., 24, 1)
    y2 = lines[..., 1:2, 1]       # (..., 24, 1)
    k = (y2 - y1)/(x2 - x1 + 1e-8)      # (..., 24, 1)
    vec = torch.cat([torch.ones_like(k, dtype=k.dtype, device=k.device), k], dim=-1)  # (..., 24, 2)
    vec = vec.unsqueeze(-2)             # (..., 24, 1, 2)
    points_ext = torch.cat([lines, points], dim=-2)         # (..., 24, 8), consider all 8 points
    den = torch.sum(points_ext * vec, dim=-1)               # (..., 24, 8) 
    proj = den / torch.norm(vec, dim=-1, keepdim=False)     # (..., 24, 8)
    proj_max = proj.max(dim=-1)[0]       # (..., 24)
    proj_min = proj.min(dim=-1)[0]       # (..., 24)
    return proj_max - proj_min

def smallest_bounding_box(corners:torch.Tensor, verbose=False):
    """return width and length of the smallest bouding box which encloses two boxes.

    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        verbose (bool, optional): If True, return area and index. Defaults to False.

    Returns:
        (torch.Tensor): width (..., 24)
        (torch.Tensor): height (..., 24)
        (torch.Tensor): area (..., )
        (torch.Tensor): index of candiatae (..., )
    """
    lines, points, _, _ = gather_lines_points(corners)
    proj = point_line_projection_range(lines, points)   # (..., 24)
    dist = point_line_distance_range(lines, points)     # (..., 24)
    area = proj * dist
    # remove area with 0 when the two points of the line have the same coordinates
    zero_mask = (area == 0).type(corners.dtype)
    fake = torch.ones_like(zero_mask, dtype=corners.dtype, device=corners.device)* 1e8 * zero_mask
    area += fake        # add large value to zero_mask
    area_min, idx = torch.min(area, dim=-1, keepdim=True)     # (..., 1)
    w = torch.gather(proj, dim=-1, index=idx)
    h = torch.gather(dist, dim=-1, index=idx)          # (..., 1)
    w = w.squeeze(-1).float()
    h = h.squeeze(-1).float()
    area_min = area_min.squeeze(-1).float()
    if verbose:
        return w, h, area_min, idx.squeeze(-1)
    else:
        return w, h

if __name__ == "__main__":
    """
    print(LINES.shape)
    print(POINTS.shape)
    print(LINES)
    print(POINTS)
    """
    from utiles import box2corners
    import matplotlib.pyplot as plt
    box1 = [0, 0, 2, 3, np.pi/6]
    box2 = [1, 5, 4, 4, -np.pi/4]
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h, a, i = smallest_bounding_box(tensor1, True)
    print("width:", w.item(), ". length:", h.item())
    print("area: ", a.item())
    print("index in 26 candidates: ", i.item())
    print("colliniear with points: ", LINES[i.item()])
    plt.scatter(corners1[:, 0], corners1[:, 1])
    plt.scatter(corners2[:, 0], corners2[:, 1])
    for i in range(corners1.shape[0]):
        plt.text(corners1[i, 0], corners1[i, 1], str(i))
    for i in range(corners2.shape[0]):
        plt.text(corners2[i, 0], corners2[i, 1], str(i+4))
    plt.axis("equal")
    plt.show()