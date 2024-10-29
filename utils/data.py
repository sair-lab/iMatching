import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch2cv(imgs):
    if len(imgs.shape) == 4:
        return np.stack([img.detach().cpu().numpy()[..., ::-1] for img in imgs.permute(0, 2, 3, 1)])


def cv2torch(imgs):
    if len(imgs.shape) == 4:
        return torch.from_numpy(imgs[..., ::-1].copy().transpose(0, 3, 1, 2))
    elif len(imgs.shape) == 3:
        return torch.from_numpy(imgs[..., ::-1].copy().transpose(2, 0, 1))

def coord_list_grid_sample(values, points, mode='bilinear'):
    dim = len(points.shape)
    points = points.view(values.size(0), 1, -1, 2) if dim == 3 else points
    output = F.grid_sample(values, points, mode, align_corners=True).permute(0, 2, 3, 1)
    return output.squeeze(1) if dim == 3 else output