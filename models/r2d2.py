from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import numpy as np

from ext.r2d2.extract import extract_multiscale, load_network, NonMaxSuppression
from ext.r2d2.nets.patchnet import Quad_L2Net_ConfCFS
from utils.config import hydra_config, hydra_instantiable

MODULE_HOME = Path(__file__).parent / "../ext/r2d2"


@hydra_config(name="r2d2_cfg_default", group="model/keypoint/config")
@dataclass
class R2D2Config:
    reliability_thr: float = 0.7
    repeatability_thr: float = 0.7
    n_features: int = 300
    model_name: str = "WASF_N16"


@hydra_instantiable(name="r2d2", group="model/keypoint")
class R2D2(nn.Module):
    def __init__(self, config: R2D2Config = R2D2Config()):
        super().__init__()
        self.model = load_network(MODULE_HOME / f"models/r2d2_{config.model_name}.pt")
        # self.model = Quad_L2Net_ConfCFS(kw=args)
        self.detector = NonMaxSuppression(
            rel_thr=config.reliability_thr,
            rep_thr=config.repeatability_thr)
        self.n_features = config.n_features

    def forward(self, img):
        fea_xy = torch.full((len(img), self.n_features, 2), np.nan, device=img.device)
        descriptors = torch.full((len(img), self.n_features, 128), np.nan, device=img.device)
        n_fea = []
        res = self.model([img])
        # get output and reliability map
        reliability = res['reliability'][0]
        repeatability = res['repeatability'][0]
        for i, (rel, rep, desc_map) in enumerate(zip(reliability, repeatability, res['descriptors'][0])):
            y, x = self.detector([rel[None]], [rep[None]])
            score = rel[0,y,x] * rep[0,y,x]
            idx = score.topk(self.n_features).indices if len(score) > self.n_features else torch.arange(len(score), device=score.device)
            n_acutal = len(idx)
            fea_x, fea_y = x[idx][:n_acutal], y[idx][:n_acutal]

            fea_xy[i, :n_acutal, 0], fea_xy[i, :n_acutal, 1] = fea_x, fea_y
            descriptors[i, :n_acutal] = desc_map[:, fea_y, fea_x].T
            n_fea.append(len(idx))

        return fea_xy, descriptors, torch.tensor(n_fea).to(img.device)
