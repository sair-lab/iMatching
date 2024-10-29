from functools import partial
from typing import Any, Sequence
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import kornia.geometry as kg

from evaluation.metrics.reprojection import mean_matching_accuracy

from .base import EvaluatorBase
from .hpatches_matching import MeanMatchingAccuracyH


class HomographyEstimationAccuracy(MeanMatchingAccuracyH):

    def __init__(self, thresh=[1.], ransac_thresh=1., solver="cv2"):
        super().__init__()
        if solver not in ("cv2", "pydegensac"):
            raise ValueError(f"Unrecognized homography solver: {solver}")
        self.solver = solver
        self.thresh = thresh
        self.ransac_thres = ransac_thresh
        # self.add_state("cum_inlier_rate", default=0., dist_reduce_fx="sum")
        # self.add_state("n_pairs", default=0., dist_reduce_fx="sum")

    def update(self, coord_src: torch.Tensor, coord_dst: torch.Tensor, h_gt: torch.Tensor, w: int, h: int):
        B = len(coord_src)
        h_pred, inliers_rate = self.estimate_homography(coord_src, coord_dst)
        corners = torch.tensor([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]).to(coord_src).repeat(B)
        corners_dst = MeanMatchingAccuracyH.reproject(corners, h_gt)
        corners_dst_pred = MeanMatchingAccuracyH.reproject(corners, h_pred)
        self.error.append(torch.norm(corners_dst_pred - corners_dst, dim=-1))
        # self.cum_inlier_rate += inliers_rate.sum()
        # self.n_pairs += B

    def estimate_homography(self, coord_src: torch.Tensor, coord_dst: torch.Tensor):
        if self.solver == "cv2":
            find_homography = partial(cv2.findHomography, method=cv2.RANSAC)
        else:
            import pydegensac
            find_homography = pydegensac.findHomography

        h_pred, inliers_rate = [], []
        coord_src_arr = coord_src.detach().cpu().numpy()
        coord_dst_arr = coord_dst.detach().cpu().numpy()
        for coord_src_arr_, coord_dst_arr_ in zip(coord_src_arr, coord_dst_arr):
            try:
                h_pred_, inliers_ = find_homography(coord_src_arr_, coord_dst_arr_, self.ransac_thres)
            except:
                h_pred_ = torch.full((3, 3), np.nan).to(coord_src)
                inliers_ = torch.zeros(len(coord_src_arr_)).to(coord_src)
            h_pred.append(h_pred_)
            inliers_rate.append(inliers_.mean())
        return torch.stack(h_pred), torch.stack(inliers_rate)


class HPatchesEvaluator(EvaluatorBase[HomographyEstimationAccuracy]):
    def __init__(self, module: pl.LightningModule, thresh: Sequence[float]) -> None:
        super().__init__(module)
        self.thresh = sorted(thresh)

    def create_matric(self):
        return HomographyEstimationAccuracy(thresh=self.thresh)

    def get_dataloader_name(self, batch: Any) -> str:
        _, ((change, _), _) = batch
        return change[0][0]

    def step_impl(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        (image_src, image_dst, homography), _ = batch

        def infer_correspondence(src, dst):
            coord_src, _, n_points = self.module.keypoint_model(src)
            _, coord_dst, _ = self.module.matching_model(xy_r=coord_src, n_r=n_points, aux_r=src, aux_n=dst)
            return coord_src, coord_dst

        # dimensions may not match for v_* sequences
        if all(img.shape == image_src[0].shape for img in image_src) \
                and all(img.shape == image_dst[0].shape for img in image_dst):
            coord_src, coord_dst = infer_correspondence(torch.stack(image_src), torch.stack(image_dst))
        else:
            coord_src_list, coord_dst_list = [], []
            for image_src_, image_dst_ in zip(image_src, image_dst):
                coord_src_, coord_dst_ = infer_correspondence(torch.stack([image_src_]), torch.stack([image_dst_]))
                coord_src_list.append(coord_src_)
                coord_dst_list.append(coord_dst_)
            coord_src, coord_dst = torch.cat(coord_src_list), torch.cat(coord_dst_list)


        self.cur_metric(coord_src, coord_dst, torch.stack(homography))

    def epoch_end(self, outputs) -> None:
        homo_mma_all = [metric.compute() for metric in self.metrics]
        homo_mma_cum, homo_mma_num = 0, 0
        for i, proj_mma in enumerate(homo_mma_all):
            n_data = len(self.dataloaders[i].dataset)
            homo_mma_cum += proj_mma * n_data
            homo_mma_num += n_data
            self.module.log_dict({f"test/homography_mma/{self.dataloader_idx_name_mapping[i]}/{th}px": proj_mma[i] for i, th in enumerate(self.thresh)})
        if homo_mma_num > 0:
            proj_mma_weighted_mean = homo_mma_cum / homo_mma_num
            self.module.log_dict({f"test/homography_mma/{th}px": proj_mma_weighted_mean[i] for i, th in enumerate(self.thresh)})
