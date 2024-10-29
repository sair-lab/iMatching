#!/usr/bin/env python3

import torch
import numpy as np
from kornia.geometry import angle_to_rotation_matrix
from torch import nn
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from ext.aspanformer.src.utils.dataset import get_resized_wh, get_divisible_wh, pad_bottom_right

class ResizeLonger:
    def __init__(self, resize, df=None):
        self.resize = resize
        self.df = df

    def __call__(self, image):
        w, h = image.shape[-1], image.shape[-2]
        w_new, h_new = get_resized_wh(w, h, self.resize)
        if self.df is not None:
            w_new, h_new = get_divisible_wh(w_new, h_new, self.df)
        return T.Resize((h_new, w_new), antialias=True)(image)

class RandAug(nn.Module):
    def __init__(self, scale=1, size=[480, 640], p=[1, 0, 0, 0], pad=False):
        super().__init__()
        self.out_size = (np.array(size) * scale).round().astype(np.int32)
        self.to_tensor = T.Compose([np.array, T.ToTensor()])
        if len(size) == 2:
            self.resize = T.Resize(self.out_size.tolist(), antialias=True)
        elif len(size) == 1: # resize the longer edge
            self.resize = ResizeLonger(size[0], df=64)
        self.rand_crop = T.RandomResizedCrop(self.out_size.tolist(), scale=(0.1, 1.0), antialias=True)
        self.rand_rotate = T.RandomRotation(15, interpolation=F.InterpolationMode.BILINEAR)
        self.rand_color = T.ColorJitter(0.8, 0.8, 0.8)
        self.p = np.array(p) / sum(p)
        self.pad = pad


    def forward(self, image, K=None, depth=None, pose=None):
        if isinstance(image, Image.Image):
            in_size = np.array(image.size[::-1])
            image = self.to_tensor(image)
            depth = depth if depth is None else self.to_tensor(depth)
        elif isinstance(image, torch.Tensor):
            in_size = np.array(image.shape[1:])
        else:
            raise ValueError(f"Unknown image type : {type(image)}")

        angle = 0

        transform = np.random.choice(np.arange(len(self.p)), p=self.p)
        if transform == 1:
            trans = self.rand_crop
            i, j, h, w = T.RandomResizedCrop.get_params(image, trans.scale, trans.ratio)
            image = F.crop(image, i, j, h, w)
            depth = F.crop(depth, i, j, h, w) if depth is not None else None
            # TODO: pose

        elif transform == 2:
            trans = self.rand_rotate
            angle = T.RandomRotation.get_params(trans.degrees)
            # diagonal_in / diagonal_out
            scale_d = in_size.max() / in_size.min() * abs(np.sin(np.deg2rad(angle))) + abs(np.cos(np.deg2rad(angle)))
            crop_size = (in_size / scale_d).astype(int)
            # fill oob pix with reflection so that model can't detect rotation with boundary
            image = F.rotate(image, angle, trans.resample, trans.expand, trans.center, trans.fill)
            image = F.center_crop(image, tuple(crop_size))
            # fill oob depth with inf so that projector can mask them out
            if depth is not None and isinstance(depth, torch.Tensor):
                depth = F.rotate(depth, angle, trans.resample, trans.expand, trans.center, trans.fill)
                depth = F.center_crop(depth, tuple(crop_size))

        elif transform == 3:
            image = self.rand_color(image)

        shape_before_resize = np.array(image.shape[-2:])
        image = self.resize(image)
        shape_after_resize = np.array(image.shape[-2:])
        scale = shape_after_resize / shape_before_resize
        scale = torch.from_numpy(scale[[1, 0]])
        depth = self.resize(depth) if depth is not None else None

        if K is not None:
            K = K.clone()
            K[0:2, 0:2] @= torch.diag(scale).to(K)
            K[0:2, 2] *= (scale).to(K)
        if pose is not None:
            pose = pose @ torch.block_diag(angle_to_rotation_matrix(torch.tensor(-angle)), torch.eye(2)).to(pose)

        if self.pad:
            h_new, w_new = shape_after_resize
            pad_to = max(h_new, w_new)
            pad_to = pad_to.item() if not isinstance(pad_to, int) else pad_to
            image, _ = pad_bottom_right(image, pad_to, ret_mask=False)
            if depth is not None:
                depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)

        return image, K, depth, pose
