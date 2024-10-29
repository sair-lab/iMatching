from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from attr import dataclass

from utils.config import hydra_config, hydra_instantiable
from utils.misc import dictseq2seqdict
import torchvision.transforms as transforms

from .correspondence import Criterion


@hydra_instantiable(name="nn", group="model/matching")
class NNMatcher(nn.Module):
    def __init__(self, metric: str = "cosine") -> None:
        super().__init__()
        self.matcher_type = "descriptor-based"
        self.metric = metric

    def forward(self, xy_r, xy_n, desc_r, desc_n, n_r, n_n, **kwargs):
        out_device = xy_r.device

        coord_dst = []
        for xy_n_, desc_r_, desc_n_, n_r_, n_n_ in zip(xy_n, desc_r, desc_n, n_r, n_n):
            n_r_ = min(n_r_, n_n_)
            n_n_ = min(n_r_, n_n_)
            if self.metric == "cosine":
                sim = (desc_r_[:n_r_] / desc_r_[:n_r_].norm(dim=1, keepdim=True)) @ (desc_n_[:n_n_] / desc_n_[:n_n_].norm(dim=1, keepdim=True)).T
                match_idx = sim.max(dim=1).indices
                coord_dst.append(xy_n_[match_idx])
            else:
                raise ValueError("Unknown metric.")

        return xy_r, torch.stack(coord_dst)
