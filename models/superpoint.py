import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ext.SuperGluePretrainedNetwork.models import superpoint as sp
from utils.config import hydra_config, hydra_instantiable

MODULE_HOME = Path(__file__).parent / "../ext/SuperGluePretrainedNetwork"


@hydra_config(name="superpoint_cfg_default", group="model/keypoint/cfg")
@dataclass
class SuperPointConfig:
    descriptor_dim: int = 256
    nms_radius: int = 4
    keypoint_threshold: float = 0.003
    max_keypoints: int = 300
    remove_borders: int = 4


@hydra_instantiable(name="superpoint", group="model/keypoint")
class SuperPoint(nn.Module):
    def __init__(self,
                 nan_to_rand: bool = False,
                 cfg: SuperPointConfig = SuperPointConfig()) -> None:
        super().__init__()
        self.nan_to_rand = nan_to_rand
        self.max_keypoints = cfg.max_keypoints

        # hydra passes in omegaconf.DictConfig instead of dataclass
        self.model = sp.SuperPoint(dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg)
        self.convert_gray = transforms.Grayscale()

    def forward(self, img):
        B = len(img)
        if self.nan_to_rand:
            fea_xy = torch.randn((B, self.max_keypoints, 2), device=img.device)
            desc = torch.randn((B, self.max_keypoints, self.model.config['descriptor_dim']), device=img.device)
        else:
            desc = torch.full((B, self.max_keypoints, self.model.config['descriptor_dim']), np.nan, device=img.device)
            fea_xy = torch.full((B, self.max_keypoints, 2), np.nan, device=img.device)

        outs = self.model({'image': self.convert_gray(img)})
        n_fea = []
        for i, (fea, des) in enumerate(zip(outs['keypoints'], outs['descriptors'])):
            fea_xy[i, :len(fea)], desc[i, :len(des.T)] = fea, des.T
            n_fea.append(self.max_keypoints if self.nan_to_rand else len(fea))

        return fea_xy, desc, torch.tensor(n_fea).to(img.device)
