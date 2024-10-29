from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from attr import dataclass

from ext.caps.CAPS.criterion import CtoFCriterion as CtoFCriterion
from ext.caps.CAPS.network import CAPSNet
from utils.config import hydra_config, hydra_instantiable
from utils.misc import dictseq2seqdict
import torchvision.transforms as transforms

from .correspondence import Criterion


# used in conjunction with ext.caps.CAPS.criterion.CtoFCriterion
@hydra_config(name="caps_ctof_cfg_default", group="model/cfg/criterion/args")
@dataclass
class CAPSCriterionConfig:
    std: bool = True
    w_epipolar_coarse: float = 1.
    w_epipolar_fine: float = 1.
    w_cycle_coarse: float = 0.1
    w_cycle_fine: float = 0.1
    w_std: float = 0.
    th_cycle: float = 0.025
    th_epipolar: float = 0.5
    window_size: float = 0.125


@hydra_instantiable(name="caps_ctof", group="model/cfg/criterion", defaults=[{"args": "caps_ctof_cfg_default"}])
class CAPSCriterion(CtoFCriterion, Criterion):

    def __init__(self, args: CAPSCriterionConfig = CAPSCriterionConfig()):
        super().__init__(args)


@hydra_config(name="capsnet_cfg_default", group="model/matching/capsnet_config")
@dataclass
class CAPSNetConfig:
    pretrained: bool = True
    backbone: str = "resnet50"
    coarse_feat_dim: int = 128
    fine_feat_dim: int = 128
    prob_from: str = "correlation"
    window_size: float = 0.125
    use_nn: bool = True


@hydra_instantiable(name="caps", group="model/matching")
class CAPS(nn.Module):
    def __init__(self,
                 capsnet_config: CAPSNetConfig = CAPSNetConfig(),
                 model_path: Optional[str] = None,
                 ransac_tol: float = -1.) -> None:
        super().__init__()
        self.matcher_type = "regression"
        
        self.model = CAPSNet(capsnet_config, "cpu")
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu")['state_dict'])
        self.ransac_tol = ransac_tol

        self.transform = transforms.Compose([
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                            std=(0.229, 0.224, 0.225)),])

    def forward(self, xy_r=None, n_r=None, img_r=None, img_n=None, apply_mask=True, **kwargs):
        img_r = self.transform(img_r)
        img_n = self.transform(img_n)

        aux_info = self.model(img_r, img_n, xy_r)
        xy_n_matched = aux_info['coord2_ef']  # [w, h]
        # xy_n_matched, std = self.model.test(img_r, img_n, xy_r)
        # aux_info = {'std': std}
        n_corr = xy_n_matched.shape[1]
        out_device = xy_n_matched.device

        # ransac
        match_idx_half = torch.arange(n_corr).to(xy_r.device)
        match_idx = match_idx_half.expand(2, n_corr).T # [[0, 0], [1, 1], ...]
        match_indices, match_masks = [], []
        for src_xy, dst_xy, n_valid in zip(xy_r, xy_n_matched, n_r):
            if self.ransac_tol > 0:
                if n_valid >= 4:
                    _, mask = cv2.findHomography(src_xy[:n_valid].detach().cpu().numpy(), dst_xy[:n_valid].detach().cpu().numpy(),
                        method=cv2.RANSAC, ransacReprojThreshold=self.ransac_tol)
                else:
                    mask = np.ones((n_valid, 1), dtype=bool)
                match_mask = torch.zeros(n_corr, dtype=bool).to(out_device)
                match_mask[:n_valid] = torch.from_numpy(mask.flatten().astype(bool)).to(out_device)
            else:
                match_mask = torch.ones(n_corr, dtype=bool).to(out_device)

            if apply_mask:
                match_indices.append(match_idx[match_mask] if apply_mask else match_idx)
            else:
                match_indices.append(match_idx)
                match_masks.append(match_mask)

        if apply_mask:
            return match_indices, xy_n_matched, dictseq2seqdict(aux_info)
        else:
            return match_indices, match_masks, (xy_n_matched, aux_info)
        # # !
        # from matplotlib import pyplot as plt
        # import numpy as np
        # indices = np.concatenate([np.arange(n) for n in n_n])
        # indices = np.stack([indices, indices]).T
        # viz_2d.showmatch(img_r, xy_r, img_n, xy_n_matched, torch.tensor(indices), torch.tensor(n_n))
        # # plt.show()
        # # !
        # xy_n = xy_n_matched
        # dist = torch.cdist(xy_n_matched, xy_n)
        # min_dist, idx = dist.min(dim=2)
        # to_merge = min_dist < self.merge_thresh
        # indices = [torch.stack((torch.arange(xy_r.shape[1]).to(i.device), i)).T[m] for m, i in zip(to_merge, idx)]
        # return indices, xy_n
        