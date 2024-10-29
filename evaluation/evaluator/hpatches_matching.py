from typing import Any, Sequence
import pytorch_lightning as pl
import torch
import torchmetrics
import kornia.geometry as kg

from .base import EvaluatorBase


class MeanMatchingAccuracyH():

    def __init__(self, thresh=1.):
        super().__init__(thresh)

    def update(self, coord_src: torch.Tensor, coord_dst: torch.Tensor, homography: torch.Tensor):
        coord_dst_pred = self.reproject(coord_src, homography)
        reproj_error = torch.norm(coord_dst_pred - coord_dst, dim=-1)
        self.error.append(reproj_error)

    @staticmethod
    def reproject(coord_src: torch.Tensor, homography: torch.Tensor):
        coord_src_h = kg.convert_points_to_homogeneous(coord_src)
        coord_dst_h = (homography @ coord_src_h.transpose(-1, -2)).transpose(-1, -2)
        return kg.convert_points_from_homogeneous(coord_dst_h)


class HPatchesMatchingEvaluator(EvaluatorBase[MeanMatchingAccuracyH]):
    def __init__(self, module: pl.LightningModule, thresh: Sequence[float]) -> None:
        super().__init__(module)
        self.thresh = sorted(thresh)

    def create_matric(self):
        return MeanMatchingAccuracyH(thresh=self.thresh)

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
            # self.module.log(f"test/homography_mma/{self.dataloader_idx_name_mapping[i]}",
            #          {f"{th}px": proj_mma[i] for i, th in enumerate(self.thresh)})
        if homo_mma_num > 0:
            proj_mma_weighted_mean = homo_mma_cum / homo_mma_num
            self.module.log_dict({f"test/homography_mma/{th}px": proj_mma_weighted_mean[i] for i, th in enumerate(self.thresh)})
