import abc
from typing import Any, Generic, Sequence, Tuple, TypeVar, Union

import kornia.geometry as kg
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.datasets.wrappers import transpose
from evaluation.evaluator.pose_based import infer_correspondence

from utils.geometry import Projector, distort_denormalize, undistort_normalize, fundamental_matrix, invert_Rt
from utils.misc import pad_to_same

from .base import EvaluatorBase


class MMA(torchmetrics.Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(self, thresh=1.):
        super().__init__()
        self.add_state("error", default=[], dist_reduce_fx="cat")
        self.thresh = thresh

    def update(self, error: torch.Tensor):
        per_sample_acc = self.matching_accuracy(error, self.thresh)
        self.error.append(per_sample_acc)

    def compute(self, thresh=None):
        if thresh is None:
            thresh = self.thresh

        return self.get_error_all().mean(0)

    def get_error_all(self):
        if len(self.error) == 0:
            return torch.zeros(0)
        elif type(self.error) is torch.Tensor:
            return self.error
        else:
            try:
                return torch.cat(self.error)
            except:
                all_error = []
                for error in self.error:
                    for i in error:
                        all_error.append(i) 
                all_error, _ = pad_to_same(all_error)
                return all_error                

    @staticmethod
    def matching_accuracy(errors: torch.Tensor, thresh: Union[float, Sequence[float]]):
        thresh_ = torch.tensor(thresh).to(errors.device)
        thresh_ndim = thresh_.ndim
        error = errors[(...,) + (None,) * thresh_ndim]

        acc = (error < thresh_[None, None]).sum(dim=1) / error.isfinite().sum(dim=1).clamp(min=1)
        return acc


class DistMetricBase(abc.ABC):

    def __init__(self, name: str) -> None:
        self.name = name

    def project(self, coord_src: torch.Tensor, *geom_aux: torch.Tensor):
        return coord_src

    def error(self, coord_src: torch.Tensor, coord_dst: torch.Tensor, *geom_aux: torch.Tensor):
        coord_dst_pred = self.project(coord_src, *geom_aux)
        if coord_dst_pred.shape != coord_dst.shape and coord_dst_pred.shape[0] == 1:
            coord_dst_pred = coord_dst_pred[:, min(coord_dst_pred.shape[1], coord_dst.shape[1])]
            coord_dst = coord_dst[:, min(coord_dst_pred.shape[1], coord_dst.shape[1])]
        return torch.norm(coord_dst_pred - coord_dst, dim=-1)


PROJ_TY = TypeVar("PROJ_TY", bound=DistMetricBase)


class DistMetricEvaluator(EvaluatorBase[MMA], Generic[PROJ_TY]):

    def __init__(self, module: pl.LightningModule, projector: PROJ_TY, thresh: Sequence[float]) -> None:
        self.thresh = sorted(thresh)
        super().__init__(module, f"{projector.name}_mma_{min(self.thresh)}px")
        self.projector = projector

    def create_matric(self):
        return MMA(thresh=self.thresh)

    def step_impl(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # geom_aux: tuple of depth, pose, homography, etc.
        (image, *geom_aux), _ = batch

        if isinstance(image, torch.Tensor):
            # tensor of shape (B, 2, ...)
            image_src, image_dst = image.swapdims(0, 1)
            coord_src, coord_dst = infer_correspondence(self.module, image_src, image_dst)
        else:
            # lists of tensors, e.g. hpatches
            image_src, image_dst = image
            if all(img.shape == image_src[0].shape for img in image_src) \
                and all(img.shape == image_dst[0].shape for img in image_dst):
                # lists of tensors of the same shape (B, ...)
                coord_src, coord_dst = infer_correspondence(self.module, torch.stack(image_src), torch.stack(image_dst))
            else:
                # lists of tensors of the different shape
                coord_src_list, coord_dst_list = [], []
                for image_src_, image_dst_ in zip(image_src, image_dst):
                    coord_src_, coord_dst_ = infer_correspondence(self.module, torch.stack([image_src_]), torch.stack([image_dst_]))
                    coord_src_list.append(coord_src_)
                    coord_dst_list.append(coord_dst_)
                coord_src, coord_dst = torch.cat(coord_src_list), torch.cat(coord_dst_list)

        # each item is of shape (B, 2, ...)
        # e.g. [depth_src, depth_dst, pose_src, pose_dst, K_src, K_dst]
        unpacked_aux = [src_or_dst for aux in geom_aux for src_or_dst in transpose(aux)]
        # _, _, pose_src, pose_dst, K_src, K_dst = unpacked_aux
        # F_mat = kg.fundamental_from_projections(K_src @ invert_Rt(pose_src)[:, 0:3, 0:4], K_dst @ invert_Rt(pose_dst)[:, 0:3, 0:4])
        if coord_src.numel() == 0 or coord_dst.numel() == 0:
            return
        self.cur_metric(self.projector.error(coord_src, coord_dst, *unpacked_aux))

    def epoch_end(self) -> None:
        proj_prefix = self.projector.name
        mma_all = self.metric_weighted_mean()
        if mma_all is not None:
            self.module.log_dict({f"{proj_prefix}_mma/{th}px": mma_all[i] for i, th in enumerate(self.thresh)}, sync_dist=True)
            self.log_stage_metric("", {f"{proj_prefix}_mma/{th}px": mma_all[i] for i, th in enumerate(self.thresh)})
            # monitor metric for checkpointing and early termination
            self.log_monitor(mma_all[0])

            # log reprojection error distribution
            if self.logger is not None and isinstance(self.logger, TensorBoardLogger):
                error_all = torch.cat([metric.get_error_all().flatten() for metric in self.metrics])
                self.logger.experiment.add_histogram(self.format_stage_metric_name(f"log_{proj_prefix}_err_hist"),
                                                     error_all[error_all.isfinite()].log10(),
                                                     global_step=self.module.global_step)

            # monitor metric for checkpointing and early termination
            # self.module.log(f"monitor/{metric_prefix}_mma_{self.thresh[0]}px", mma_all[0])


class EpipolarDist(DistMetricBase):

    def __init__(self):
        super().__init__("epipolar")

    def error(self, coord_src: torch.Tensor, coord_dst: torch.Tensor, *geom_aux: torch.Tensor):
        if len(geom_aux) == 6:
            _, _, pose_src, pose_dst, K_src, K_dst = geom_aux
        elif len(geom_aux) == 4:
            pose_src, pose_dst, K_src, K_dst = geom_aux
        else:
            raise ValueError("Must be [depth_src, depth_dst,] pose_src, pose_dst, K_src, K_dst")
        F_mat = fundamental_matrix(pose_src, K_src, pose_dst, K_dst)

        return kg.sampson_epipolar_distance(coord_src, coord_dst, F_mat)


class EpipolarDistEvaluator(DistMetricEvaluator[EpipolarDist]):

    def __init__(self, module: pl.LightningModule, thresh: Sequence[float]) -> None:
        super().__init__(module, EpipolarDist(), thresh)


class ReprojectionDist(DistMetricBase):

    def __init__(self):
        super().__init__("reproj")

    def project(self, coord_src: torch.Tensor, *geom_aux: torch.Tensor):
        depth_src, depth_dst, pose_src, pose_dst, K_src, K_dst = geom_aux
        _, _, H, W = depth_src.shape

        # extract depth per point from coord_src
        row_idx = coord_src[:, :, 1].long()
        col_idx = coord_src[:, :, 0].long()

        row_idx = torch.clamp(row_idx, min=0, max=H-1)
        col_idx = torch.clamp(col_idx, min=0, max=W-1)

        nan_mask = coord_src.isnan().any(dim=2)
        # ravel index
        rav_idx = row_idx * W + col_idx
        rav_idx[nan_mask] = 0
        point_depth = depth_src.flatten(-3, -1).gather(-1, rav_idx)
        point_depth[nan_mask] = -1
        invalid_depth = point_depth < 1e-4

        coord_src_normed = undistort_normalize(coord_src, K_src)
        points_w = Projector.pix2world(coord_src_normed, depth_src, invert_Rt(pose_src), K_src)
        coord_dst_pred, _ = Projector.world2pix(points_w, (H, W), invert_Rt(pose_dst), K_dst)
        coord_dst_pred = distort_denormalize(coord_dst_pred, K_dst)

        coord_dst_pred[invalid_depth] = torch.nan
        return coord_dst_pred


class ReprojectionDistEvaluator(DistMetricEvaluator[ReprojectionDist]):

    def __init__(self, module: pl.LightningModule, thresh: Sequence[float]) -> None:
        super().__init__(module, ReprojectionDist(), thresh)
