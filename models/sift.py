import dataclasses
from dataclasses import dataclass
from pathlib import Path
import cv2

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from utils.config import hydra_config, hydra_instantiable

MODULE_HOME = Path(__file__).parent / "../ext/SuperGluePretrainedNetwork"


@hydra_instantiable(name="sift", group="model/keypoint")
class Sift(nn.Module):
    def __init__(self,
                 nan_to_rand: bool = False,
                 max_keypoints=1300) -> None:
        super().__init__()
        self.nan_to_rand = nan_to_rand
        self.max_keypoints = max_keypoints
        self.model = cv2.SIFT_create(nfeatures=max_keypoints)
        self.convert_gray = transforms.Grayscale()
        self.descriptor_dim = 128

    def forward(self, img):
        B = len(img)
        if self.nan_to_rand:
            fea_xy = torch.randn((B, self.max_keypoints, 2), device=img.device)
            desc = torch.randn((B, self.max_keypoints, self.descriptor_dim), device=img.device)
        else:
            desc = torch.full((B, self.max_keypoints, self.descriptor_dim), np.nan, device=img.device)
            fea_xy = torch.full((B, self.max_keypoints, 2), np.nan, device=img.device)
        gray = self.convert_gray(img).cpu().numpy() * 255
        gray = gray.astype(np.uint8)
        n_fea = []
        for i in range(B):
            kpts = self.model.detect(gray[i][0])
            kpts, feats = self.model.compute(gray[i][0], kpts)
            n = 0
            for j in range(min(len(kpts), self.max_keypoints)):
                fea_xy[i][j] = torch.tensor(kpts[j].pt)
                desc[i][j] = torch.tensor(feats[j])
                n += 1
            n_fea.append(n)

        return fea_xy, desc, torch.tensor(n_fea).to(img.device)
