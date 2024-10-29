from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from attr import dataclass

from ext.SuperGluePretrainedNetwork.models.matching import SuperPoint, SuperGlue
from utils.config import hydra_config, hydra_instantiable
from utils.misc import dictseq2seqdict


@hydra_config(name="superpoint_superglue_cfg_default", group="model/matching/config")
@dataclass
class SuperPointSuperGlueConfig:
    sp_nms_radius: int = 4
    sp_keypoint_threshold: float = 0.003
    sp_max_keypoints: int = 300
    sg_weights: str = 'indoor'
    sg_sinkhorn_iterations: int = 100
    sg_match_threshold: float = 0.2


@hydra_instantiable(name="superpoint_superglue", group="model/matching")
class SuperPointSuperGlue(nn.Module):
    def __init__(self, config: SuperPointSuperGlueConfig = SuperPointSuperGlueConfig()) -> None:
        super().__init__()
        self.matcher_type = "detector-free"

        config_dict = {
            'superpoint': {
                'descriptor_dim': 256,
                'nms_radius': config.sp_nms_radius,
                'keypoint_threshold': config.sp_keypoint_threshold,
                'max_keypoints': config.sp_max_keypoints,
                'remove_borders': 4,
            },
            'superglue': {
                'descriptor_dim': 256,
                'weights': config.sg_weights,
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': config.sg_sinkhorn_iterations,
                'match_threshold': config.sg_match_threshold
            }
        }

        self.convert_gray = transforms.Grayscale()

        self.superpoint = SuperPoint(config_dict.get('superpoint', {}))
        self.superglue = SuperGlue(config_dict.get('superglue', {}))
        self.nfeatures = config.sp_max_keypoints

    def forward(self, xy_r=None, n_r=None, img_r=None, img_n=None, apply_mask=True, **kwargs):
        data = {'image0': self.convert_gray(img_r), 'image1': self.convert_gray(img_n)}

        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = list(data[k])
                max_len = max(d.T.shape[0] for d in data[k]) if 'desc' in k else max(d.shape[0] for d in data[k])
                for i, d in enumerate(data[k]):
                    data_ki = data[k][i].T if 'desc' in k else data[k][i]
                    data[k][i] = torch.full((max_len,) + data_ki.shape[1:], 0).to(data_ki)
                    data[k][i][:len(data_ki)] = data_ki
                    data[k][i] = data[k][i].T if 'desc' in k else data[k][i]
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        match_indices = []
        n_corrs = []
        xy_rn_matched = torch.full((len(img_r), self.nfeatures, 4), np.nan, device=img_r.device)
        for src_xy, dst_xy, idx in zip(data['keypoints0'], data['keypoints1'], pred['matches0']):
            match_idx = np.array([(i, m) for i, m in enumerate(idx.cpu().numpy()) if m != -1])

            if len(match_idx) > 0:
                valid_corr = torch.cat([src_xy[match_idx[:, 0]], dst_xy[match_idx[:, 1]]], dim=1)
            else:
                valid_corr = torch.zeros(0, 4)

            match_indices.append(match_idx)
            xy_rn_matched[i, :len(valid_corr)] = valid_corr
            n_corrs.append(len(valid_corr))

        return match_indices, xy_rn_matched, n_corrs, [None for _ in range(len(img_r))]
