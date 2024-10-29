from typing import Any, Sequence
import pytorch_lightning as pl
import torch
import torchmetrics
from .projection_based import ReprojectionDist

from utils.visualization.multi import MultiVisualizer

from .base import EvaluatorBase


class TartanAirMatchVisualEvaluator(EvaluatorBase):
    def __init__(self, module: pl.LightningModule, thresh: float) -> None:
        super().__init__(module)
        self.thresh = thresh
        self.viz = MultiVisualizer(display="gif", save_dir="output/visualization", framerate=1.5)
        self.projector = ReprojectionDist()

    def create_matric(self):
        return None

    def get_dataloader_name(self, batch: Any) -> str:
        _, (series, _) = batch
        return "_".join([s[0][0] for s in series])

    def step_impl(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        (image, depth, pose, K), _ = batch
        # each item is of shape (B, 2, ...)
        image_src, image_dst = image.swapdims(0, 1)
        depth_src, depth_dst = depth.swapdims(0, 1)
        pose_src, pose_dst = pose.swapdims(0, 1)
        K_src, K_dst = K.swapdims(0, 1)

        coord_src, _, n_points = self.module.keypoint_model(image_src)
        _, coord_dst, _ = self.module.matching_model(xy_r=coord_src, n_r=n_points, aux_r=image_src, aux_n=image_dst)

        coord_dst_pred = self.projector.project(coord_src, depth_src, depth_dst, pose_src, pose_dst, K_src, K_dst)
        reproj_error = torch.norm(coord_dst_pred - coord_dst, dim=-1)

        name = self.cur_dataloader_name
        for i_src, c_src, i_dst, c_dst, err in zip(image_src, coord_src, image_dst, coord_dst, reproj_error):
            values = (err / self.thresh).log2()
            m_sizes = [len(c_dst)]
            self.viz.showmatch(i_src[None], c_src[None], i_dst[None], c_dst[None], m_sizes=m_sizes, name=name, color="coolwarm", values=values, pts_colors=('white', 'white'), vlim=[-1, 1], step=batch_idx, alpha=0.7)

    def epoch_end(self, outputs) -> None:
        self.viz.close()
