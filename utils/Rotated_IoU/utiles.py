'''
numpy implementation of 2d box intersection with test cases. 
to demonstrate the idea and validate the result of torch implementation

author: lanxiao li
2020.8
'''
import numpy as np
import matplotlib.pyplot as plt
EPSILON = 1e-8

def line_seg_intersection(line1:np.array, line2:np.array):
    """find intersection of 2 lines defined by their end points

    Args:
        line1 (np.array): (2, 2), end points of line
        line2 (np.array): (2, 2), end points of line
    Returns:
        intersection: coordinte of intersection point. None is not exists.
    """
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    assert line1.shape == (2,2)
    assert line2.shape == (2,2)
    x1, y1 = line1[0,:]
    x2, y2 = line1[1,:]
    x3, y3 = line2[0,:]
    x4, y4 = line2[1,:]
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if np.abs(num) < EPSILON:
        return None
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    if t < 0 or t > 1:
        return None
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = - den_u / num
    if u < 0 or u > 1:
        return None
    
    return [x1+t*(x2-x1), y1+t*(y2-y1)]

def box2corners(x, y, w, h, alpha):
    """
    box parameters to four box corners
    """
    x4 = np.array([0.5, -0.5, -0.5, 0.5]) * w
    y4 = np.array([0.5, 0.5, -0.5, -0.5]) * h
    corners = np.stack([x4, y4], axis=1)
    sin = np.sin(alpha)
    cos = np.cos(alpha)
    R = np.array([[cos, -sin],[sin, cos]])
    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated

def box_intersection(corners1, corners2):
    """find intersection points pf two boxes

    Args:
        corners1 (np.array): 4x2 coordinates of corners
        corners2 (np.array): 4x2 coordinates of corners
    Returns:
        inters (4, 4, 2): (i, j, :) means intersection of i-th edge of box1 with j-th of box2
        mask (4, 4) bool: (i, j) indicates if intersection exists 
    """
    assert corners1.shape == (4,2) 
    assert corners2.shape == (4,2)
    inters = np.zeros([4,4,2]) # edges of box1, edges of box2, coordinates
    mask = np.zeros([4,4]).astype(np.bool)
    for i in range(4):
        line1 = np.stack([corners1[i, :], corners1[(i+1)%4, :]], axis=0)
        for j in range(4):
            line2 = np.stack([corners2[j, :], corners2[(j+1)%4, :]], axis=0)
            it = line_seg_intersection(line1, line2)
            if it is not None:
                inters[i, j, :] = it
                mask[i, j] = True
    return inters, mask 

def point_in_box(point, corners):
    """check if a point lies in a rectangle defined by corners.
    idea: check projection

    Args:
        point (2,): coordinate of point
        corners (4, 2): coordinate of corners

    Returns:
        True if point in box
    """
    assert corners.shape == (4,2)
    a = corners[0, :]
    b = corners[1, :]
    d = corners[3, :]
    ab = b - a
    am = point - a
    ad = d - a
    # consider projection of AM on the edge AB and AD
    p_ab = np.dot(ab, am)
    norm_ab = np.dot(ab, ab)
    p_ad = np.dot(ad, am)
    norm_ad = np.dot(ad, ad)
    cond1 = p_ab > 0 and p_ab < norm_ab
    cond2 = p_ad > 0 and p_ad < norm_ad
    return cond1 and cond2

def box_in_box(corners1, corners2):
    """check if corners of 2 boxes lie in each other

    Args:
        corners1 (np.array): 4x2 coordinates of corners
        corners2 (np.array): 4x2 coordinates of corners

    Returns:
        c1_in_2 (4, ): i-th corner of box1 in box2
        c2_in_1 (4, ): i-th corner of box2 in box1 
    """
    assert corners1.shape == (4,2) 
    assert corners2.shape == (4,2)
    c1_in_2 = np.zeros((4,)).astype(np.bool)
    c2_in_1 = np.zeros((4,)).astype(np.bool)
    for i in range(4):
        if point_in_box(corners1[i, :], corners2):
            c1_in_2[i] = True
        if point_in_box(corners2[i, :], corners1):
            c2_in_1[i] = True
    return c1_in_2, c2_in_1

def intersection_poly(corners1, corners2):
    """find all vertices of the polygon for intersection of 2 boxes
    vertices include intersection points of edges and box corner in the other box

    Args:
        corners1 (np.array): 4x2 coordinates of corners
        corners2 (np.array): 4x2 coordinates of corners

    Returns:
        poly_vertices (N, 2): vertices of polygon
    """
    # corner1 = box2corners(*box1)
    # corner2 = box2corners(*box2)
    
    c1_in_2, c2_in_1 = box_in_box(corners1, corners2)
    corners_eff = np.concatenate([corners1[c1_in_2,:], corners2[c2_in_1,:]], axis=0)

    inters, mask = box_intersection(corners1, corners2)
    inters_lin = np.reshape(inters, (-1, 2))
    mask_lin = np.reshape(mask, (-1, ))
    inter_points = inters_lin[mask_lin, :]

    poly_vertices = np.concatenate([corners_eff, inter_points], axis=0)
    return poly_vertices

def compare_vertices(v1, v2):
    """compare two points according to the its angle around the origin point
    of coordinate system. Useful for sorting vertices in anti-clockwise order

    Args:
        v1 (2, ): x1, y1
        v2 (2, ): x2, y2

    Returns:
        int : 1 if angle1 > angle2. else -1
    """
    x1, y1 = v1
    x2, y2 = v2
    n1 = np.sqrt(x1*x1 + y1*y1) + EPSILON
    n2 = np.sqrt(x2*x2 + y2*y2) + EPSILON
    if y1 > 0 and y2 < 0:
        return -1
    elif y1 < 0 and y2 > 0:
        return 1
    elif y1 > 0 and y2 > 0:
        if x1/n1 < x2/n2:
            return 1
        else:
            return -1
    else:
        if x1/n1 > x2/n2:
            return 1
        else:
            return -1

import functools
def vertices2area(vertices):
    """sort vertices in anti-clockwise order and calculate the area of polygon

    Args:
        vertices (N, 2) with N>2: vertices of a convex polygon

    Returns:
        area: area of polygon
        ls: sorted vertices (normalized to centroid)
    """
    mean = np.mean(vertices, axis=0, keepdims=True)
    vertices_normalized = vertices - mean
    # sort vertices clockwise
    ls = np.array(list(sorted(vertices_normalized, key=functools.cmp_to_key(compare_vertices))))
    ls_ext = np.concatenate([ls, ls[0:1, :]], axis=0)
    total = ls_ext[0:-1, 0]*ls_ext[1:, 1] - ls_ext[1:, 0] * ls_ext[0:-1, 1]
    total = np.sum(total)
    area = np.abs(total) / 2
    return area, ls

def box_intersection_area(box1, box2):
    corners1 = box2corners(*box1)
    corners2 = box2corners(*box2)
    v = intersection_poly(corners1, corners2)
    if v.shape[0] < 3:
        return 0
    else:
        return vertices2area(v)


# --------------------------------------------------------
# tests
# --------------------------------------------------------

def test_line_seg_intersection():
    line1 = np.array([[0, 0], [0, 1]])
    line2 = np.array([[1, 0], [1, 1]])

    line3 = np.array([[0, 0], [1, 1]])
    line4 = np.array([[1, 0], [0, 1]])

    line5 = np.array([[0, 0], [1, 0]])
    line6 = np.array([[0, 1], [1, 0.5]])

    print(line_seg_intersection(line1, line2))
    print(line_seg_intersection(line3, line4))
    print(line_seg_intersection(line5, line6))

def test_box2corners():
    corners = box2corners(1, 1, 2, 3, np.pi/6)
    plt.figure()
    plt.scatter(corners[:, 0], corners[:, 1])
    for i in range(corners.shape[0]):
        plt.text(corners[i, 0], corners[i, 1], str(i))
    plt.axis("equal")
    plt.show()

    corners = box2corners(3, 1, 4, 2, np.pi/4)
    plt.figure()
    plt.scatter(corners[:, 0], corners[:, 1])
    for i in range(corners.shape[0]):
        plt.text(corners[i, 0], corners[i, 1], str(i))
    plt.axis("equal")
    plt.show()

def test_box_intersection(box1, box2):
    corners1 = box2corners(*box1)
    corners2 = box2corners(*box2)
    inters, mask = box_intersection(corners1, corners2)
    num_inters = np.sum(mask.astype(np.int))
    inters_lin = np.reshape(inters, (-1, 2))
    mask_lin = np.reshape(mask, (-1, ))
    inter_points = inters_lin[mask_lin, :]
    print("find %d intersections"%num_inters)

    corners1 = box2corners(*box1)
    corners2 = box2corners(*box2)
    plt.figure()
    plt.scatter(corners1[:, 0], corners1[:, 1])
    plt.scatter(corners2[:, 0], corners2[:, 1])
    plt.scatter(inter_points[:, 0], inter_points[:, 1], marker='x')
    for i in range(corners1.shape[0]):
        plt.text(corners1[i, 0], corners1[i, 1], str(i))
    for i in range(corners2.shape[0]):
        plt.text(corners2[i, 0], corners2[i, 1], str(i))
    plt.axis("equal")
    plt.show()

def test_point_in_box():
    p = np.random.rand(5000, 2)
    p[:, 0] *= 12
    p[:, 0] -= 5
    p[:, 1] *= 12
    p[:, 1] -= 5
    corners = box2corners(3, 1, 4, 2, np.pi/4)
    plt.figure()
    plt.scatter(corners[:, 0], corners[:, 1])
    for i in range(corners.shape[0]):
        plt.text(corners[i, 0], corners[i, 1], str(i))
    mask = [point_in_box(x, corners) for x in p]
    plt.scatter(p[mask, 0], p[mask, 1], marker="x")
    plt.axis("equal")
    plt.show()

def test_intersection_area(box1, box2):
    area, corners = box_intersection_area(box1, box2)
    print(area) 
    print(corners)
    plt.figure()
    plt.scatter(corners[:, 0], corners[:, 1])
    for i in range(corners.shape[0]):
        plt.text(corners[i, 0], corners[i, 1], str(i))
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    # test_line_seg_intersection()
    # test_box2corners()
    # test_point_in_box()
    
    box1 = np.array([0, 0, 2, 3, np.pi/6])
    box2 = np.array([1, 1, 4, 4, -np.pi/4])
    test_box_intersection(box1, box2)
    test_intersection_area(box1, box2)
