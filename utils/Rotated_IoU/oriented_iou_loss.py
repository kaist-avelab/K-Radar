import torch
import numpy as np
try:
    from box_intersection_2d import oriented_box_intersection_2d
    from min_enclosing_box import smallest_bounding_box
except:
    from utils.Rotated_IoU.box_intersection_2d import oriented_box_intersection_2d
    from utils.Rotated_IoU.min_enclosing_box import smallest_bounding_box

def box2corners_th(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

def cal_iou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u

def cal_diou(box1:torch.Tensor, box2:torch.Tensor, enclosing_type:str="smallest"):
    """calculate diou loss

    Args:
        box1 (torch.Tensor): [description]
        box2 (torch.Tensor): [description]
    """
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    c2 = w*w + h*h      # (B, N)
    x_offset = box1[...,0] - box2[..., 0]
    y_offset = box1[...,1] - box2[..., 1]
    d2 = x_offset*x_offset + y_offset*y_offset
    diou_loss = 1. - iou + d2/c2
    return diou_loss, iou

def cal_giou(box1:torch.Tensor, box2:torch.Tensor, enclosing_type:str="smallest"):
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    area_c =  w*h
    giou_loss = 1. - iou + ( area_c - u )/area_c
    return giou_loss, iou

def cal_iou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, verbose=False):
    """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
    """
    box1 = box3d1[..., [0,1,3,4,6]]     # 2d box
    box2 = box3d2[..., [0,1,3,4,6]]
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.)
    iou_2d, corners1, corners2, u = cal_iou(box1, box2)        # (B, N)
    intersection_3d = iou_2d * u * z_overlap
    v1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    v2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    u3d = v1 + v2 - intersection_3d
    if verbose:
        z_range = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.)
        return intersection_3d / u3d, corners1, corners2, z_range, u3d
    else:
        return intersection_3d / u3d

def cal_giou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, enclosing_type:str="smallest"):
    """calculated 3d GIoU loss. assume the 3d bounding boxes are only rotated around z axis

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".

    Returns:
        (torch.Tensor): (B, N) 3d GIoU loss
        (torch.Tensor): (B, N) 3d IoU
    """
    iou3d, corners1, corners2, z_range, u3d = cal_iou_3d(box3d1, box3d2, verbose=True)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    v_c = z_range * w * h
    giou_loss = 1. - iou3d + (v_c - u3d)/v_c
    return giou_loss, iou3d

def cal_diou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, enclosing_type:str="smallest"):
    """calculated 3d DIoU loss. assume the 3d bounding boxes are only rotated around z axis

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".

    Returns:
        (torch.Tensor): (B, N) 3d DIoU loss
        (torch.Tensor): (B, N) 3d IoU
    """
    iou3d, corners1, corners2, z_range, u3d = cal_iou_3d(box3d1, box3d2, verbose=True)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    x_offset = box3d1[...,0] - box3d2[..., 0]
    y_offset = box3d1[...,1] - box3d2[..., 1]
    z_offset = box3d1[...,2] - box3d2[..., 2]
    d2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset
    c2 = w*w + h*h + z_range*z_range
    diou = 1. - iou3d + d2/c2
    return diou, iou3d

def enclosing_box(corners1:torch.Tensor, corners2:torch.Tensor, enclosing_type:str="smallest"):
    if enclosing_type == "aligned":
        return enclosing_box_aligned(corners1, corners2)
    elif enclosing_type == "pca":
        return enclosing_box_pca(corners1, corners2)
    elif enclosing_type == "smallest":
        return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    else:
        ValueError("Unknow type enclosing. Supported: aligned, pca, smallest")

def enclosing_box_aligned(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the smallest enclosing box (axis-aligned)

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    w = x_max - x_min       # (B, N)
    h = y_max - y_min
    return w, h

def enclosing_box_pca(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the rotated smallest enclosing box using PCA

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    B = corners1.size()[0]
    c = torch.cat([corners1, corners2], dim=2)      # (B, N, 8, 2)
    c = c - torch.mean(c, dim=2, keepdim=True)      # normalization
    c = c.view([-1, 8, 2])                          # (B*N, 8, 2)
    ct = c.transpose(1, 2)                          # (B*N, 2, 8)
    ctc = torch.bmm(ct, c)                          # (B*N, 2, 2)
    # NOTE: the build in symeig is slow!
    # _, v = ctc.symeig(eigenvectors=True)
    # v1 = v[:, 0, :].unsqueeze(1)                   
    # v2 = v[:, 1, :].unsqueeze(1)
    v1, v2 = eigenvector_22(ctc)
    v1 = v1.unsqueeze(1)                            # (B*N, 1, 2), eigen value
    v2 = v2.unsqueeze(1)
    p1 = torch.sum(c * v1, dim=-1)                  # (B*N, 8), first principle component
    p2 = torch.sum(c * v2, dim=-1)                  # (B*N, 8), second principle component
    w = p1.max(dim=-1)[0] - p1.min(dim=-1)[0]       # (B*N, ),  width of rotated enclosing box
    h = p2.max(dim=-1)[0] - p2.min(dim=-1)[0]       # (B*N, ),  height of rotated enclosing box
    return w.view([B, -1]), h.view([B, -1])

def eigenvector_22(x:torch.Tensor):
    """return eigenvector of 2x2 symmetric matrix using closed form
    
    https://math.stackexchange.com/questions/8672/eigenvalues-and-eigenvectors-of-2-times-2-matrix
    
    The calculation is done by using double precision

    Args:
        x (torch.Tensor): (..., 2, 2), symmetric, semi-definite
    
    Return:
        v1 (torch.Tensor): (..., 2)
        v2 (torch.Tensor): (..., 2)
    """
    # NOTE: must use doule precision here! with float the back-prop is very unstable
    a = x[..., 0, 0].double()
    c = x[..., 0, 1].double()
    b = x[..., 1, 1].double()                                # (..., )
    delta = torch.sqrt(a*a + 4*c*c - 2*a*b + b*b)
    v1 = (a - b - delta) / 2. /c
    v1 = torch.stack([v1, torch.ones_like(v1, dtype=torch.double, device=v1.device)], dim=-1)    # (..., 2)
    v2 = (a - b + delta) / 2. /c
    v2 = torch.stack([v2, torch.ones_like(v2, dtype=torch.double, device=v2.device)], dim=-1)    # (..., 2)
    n1 = torch.sum(v1*v1, keepdim=True, dim=-1).sqrt()
    n2 = torch.sum(v2*v2, keepdim=True, dim=-1).sqrt()
    v1 = v1 / n1
    v2 = v2 / n2
    return v1.float(), v2.float()

if __name__ == "__main__":
    box3d1 = np.array([0,0,0,3,3,3,0])
    box3d2 = np.array([1,1,1,2,2,2,np.pi/3])
    tensor1 = torch.FloatTensor(box3d1).unsqueeze(0).unsqueeze(0).cuda()
    tensor2 = torch.FloatTensor(box3d2).unsqueeze(0).unsqueeze(0).cuda()
    giou_loss, iou = cal_giou_3d(tensor1, tensor1)
    print(giou_loss)
    print(iou)
