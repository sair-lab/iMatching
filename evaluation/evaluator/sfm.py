from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

from vo.tracking.tracker import Tracker

from .base import EvaluatorBase


class InlierRate(torchmetrics.Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("n_corr", default=[], dist_reduce_fx="cat")
        self.add_state("n_inlier", default=[], dist_reduce_fx="cat")

    def update(self, n_corr: torch.Tensor, n_inlier: torch.Tensor):
        self.n_corr.append(n_corr)
        self.n_inlier.append(n_inlier)

    def compute(self, thresh=None):
        return torch.cat(self.n_inlier).sum() / torch.cat(self.n_corr).sum()


class SfMEvaluator(EvaluatorBase[InlierRate]):

    def __init__(self, module: pl.LightningModule, vo: Tracker) -> None:
        super().__init__(module, "sfm_inlier_rate")
        self.vo = vo

    def create_matric(self):
        return InlierRate()

    def get_dataloader_name(self, batch: Any) -> str:
        _, (series_key, _) = batch
        return "_".join(part[0][0] for part in series_key)

    def step_impl(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # geom_aux: tuple of depth, pose, homography, etc.
        (images, _, _, Ks), _ = batch

        n_corr_, n_inlier_ = [], []
        for imgs, K in zip(images, Ks):
            self.vo.reset()
            self.vo.track(imgs, K)

            n_inlier = 0
            for lm in self.vo.map.pose_graph.landmarks:
                for fea in lm.track:
                    for other_fea in fea.correspondences:
                        if lm.track_contains(other_fea) and other_fea.frame_id != fea.frame_id:
                            n_inlier += 1
            n_inlier_.append(n_inlier)

            n_corr = 0
            for frame in self.vo.map.pose_graph.keyframes:
                n_corr += sum(1 for _ in frame.correspondent_features_proper)
            n_corr_.append(n_corr)

        self.cur_metric(torch.tensor(n_corr_), torch.tensor(n_inlier_))

    def epoch_end(self, outputs) -> None:
        mean_inlier_rate = self.metric_weighted_mean()
        self.module.log(f"{self.stage_name}/sfm_inlier_rate", mean_inlier_rate)
        self.log_stage_metric(f"sfm_inlier_rate", mean_inlier_rate)
        # monitor metric for checkpointing and early termination
        self.log_monitor(mean_inlier_rate)
        if self.logger is not None and isinstance(self.logger, TensorBoardLogger):
            ir_all = torch.cat([torch.cat(metric.n_inlier) / torch.cat(metric.n_corr) for metric in self.metrics])
            self.logger.experiment.add_histogram(f"{self.stage_name}/sfm_inlier_rate_hist",
                                                    ir_all[ir_all.isfinite()],
                                                    global_step=self.module.global_step)

        self.vo.reset()
