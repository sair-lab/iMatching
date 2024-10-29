import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.data import torch2cv, cv2torch
from utils.config import  hydra_instantiable


@hydra_instantiable(name="orb_matcher", group="model/matching")
class ORBMatcher(nn.Module):
    def __init__(self, nfeatures: int = 500, ransac_tol: int  = -1):
        super().__init__()
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher_create(crossCheck=False)
        self.nfeatures = nfeatures
        self.ransac_tol = ransac_tol
        self.matcher_type = "detector-free"

    def forward(self, img_r=None, img_n=None, apply_mask=True, **kwargs):
        img_r, img_n = torch2cv(img_r), torch2cv(img_n)

        match_indices = []
        match_masks = []
        n_corrs = []
        xy_rn_matched = torch.full((len(img_r), self.nfeatures, 4), np.nan)
        for i, (img_cv_r, img_cv_n) in enumerate(zip(img_r, img_n)):
            img_cv_r = cv2.cvtColor((img_cv_r * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            img_cv_n = cv2.cvtColor((img_cv_n * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            
            kp_r, desc_r = self.orb.detectAndCompute(img_cv_r, None)
            kp_n, desc_n = self.orb.detectAndCompute(img_cv_n, None)
            kp_r_array = np.array([kp_.pt for kp_ in kp_r])
            kp_n_array = np.array([kp_.pt for kp_ in kp_n])

            if len(kp_r_array) == 0 or len(kp_n_array) == 0:
                match_idx = np.zeros((0, 2), dtype=int)
            else:
                match_results = self.matcher.match(desc_n, desc_r)
                match_idx = np.array([(m.trainIdx, m.queryIdx) for m in match_results])

            if self.ransac_tol > 0 and len(match_idx) > 4:
                _, mask = cv2.findHomography(kp_r_array[match_idx[:, 0]], kp_n_array[match_idx[:, 1]],
                    method=cv2.RANSAC, ransacReprojThreshold=self.ransac_tol)
                match_mask = mask.flatten().astype(bool)
            else:
                match_mask = np.ones(len(match_idx), dtype=bool)

            if len(match_idx) > 0:
                valid_corr = np.concatenate([kp_r_array[match_idx[:, 0]], kp_n_array[match_idx[:, 1]]], axis=1)
            else:
                valid_corr = np.zeros((0, 4))

            if apply_mask:
                match_indices.append(match_idx[match_mask] if apply_mask else match_idx)
            else:
                match_indices.append(match_idx)
                match_masks.append(match_mask)

            match_indices.append(match_idx)
            xy_rn_matched[i, :len(valid_corr)] = torch.from_numpy(valid_corr).to(xy_rn_matched)
            n_corrs.append(len(valid_corr))

        return match_indices, xy_rn_matched, n_corrs, [None for _ in range(len(img_r))]
