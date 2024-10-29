import torch
import numpy as np
from torchmetrics import Metric
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Any, Generic, Sequence, Tuple, TypeVar, Union

from .base import EvaluatorBase
from dataset.datasets.wrappers import transpose
from ext.aspanformer.src.utils.metrics import estimate_pose, relative_pose_error, error_auc
from utils.geometry import Projector, distort_denormalize, undistort_normalize, fundamental_matrix, invert_Rt
class AUC(Metric):
    def __init__(self, thresh=None):
        super().__init__()
        self.angular_thresholds = [5, 10, 20] if thresh is None else thresh
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, errors):
        self.errors.append(errors.to(self.device))

    def get_error_all(self):
        if len(self.errors) == 0:
            return torch.zeros(0)
        elif type(self.errors) is torch.Tensor:
            return self.errors
        else:
            try:
                return torch.cat(self.errors)
            except:
                all_error = []
                for error in self.errors:
                    for i in error:
                        all_error.append(i) 
                all_error, _ = pad_to_same(all_error)
                return all_error      
        
    def compute(self, thresh=None):
        if thresh is None:
            thresh = self.angular_thresholds
        pose_errors = self.get_error_all().cpu().numpy()
        aucs = error_auc(pose_errors, self.angular_thresholds)
        res = []
        for t in thresh:
            res.append(aucs[f'auc@{t}'])
        res = torch.tensor(res, dtype=torch.float32)
        assert not res.isnan().any(), str(pose_errors) + '\n ------------------------------- \n' + str(aucs)
        return res
    
def compute_pose_errors(data, pixel_thr=0.5, conf=0.99999):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    # pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    # conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs, (pts0, pts1) in enumerate(zip(data['mkpts0_f'], data['mkpts1_f'])):
        pts0 = pts0.cpu().numpy()
        pts1 = pts1.cpu().numpy()
        ret = estimate_pose(pts0, pts1, K0[bs], K1[bs], pixel_thr, conf=conf)


        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.1)
            if np.isnan(t_err) or np.isnan(R_err):
                t_err = np.inf
                R_err = np.inf
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)

def infer_correspondence(module, src, dst) -> Tuple[torch.Tensor, torch.Tensor]:
    if module.matching_model.matcher_type == "regression" and module.keypoint_model is not None:
        coord_src, _, n_points = module.keypoint_model(src)
        _, coord_dst, _ = module.matching_model(xy_r=coord_src, n_r=n_points, img_r=src, img_n=dst)
    elif module.matching_model.matcher_type == "detector-free" and module.keypoint_model is None:
        _, coords, n_corrs, _ = module.matching_model(img_r=src, img_n=dst)
        coord_src, coord_dst = coords[..., :2], coords[..., 2:]
    elif module.matching_model.matcher_type == "descriptor-based" and module.keypoint_model is not None:
        coord_src, desc_r, n_points_src = module.keypoint_model(src)
        coord_dst, desc_n, n_points_dst = module.keypoint_model(dst)
        coord_src, coord_dst = module.matching_model(xy_r=coord_src, xy_n=coord_dst, desc_r=desc_r, desc_n=desc_n,
                                                            n_r=n_points_src,  n_n=n_points_dst, img_r=src, img_n=dst)
    else:
        raise ValueError("Unknown matching scheme.")
    return coord_src, coord_dst

class PoseEvaluator(EvaluatorBase[AUC]):

    def __init__(self, module: pl.LightningModule, thresh: Sequence[float]) -> None:
        self.thresh = sorted(thresh)
        super().__init__(module, f"pose_auc_{min(self.thresh)}px")

    def create_matric(self):
        return AUC(thresh=self.thresh)

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
        coord_src, coord_dst = coord_src.type(torch.float32), coord_dst.type(torch.float32)
        # each item is of shape (B, 2, ...)
        # e.g. [depth_src, depth_dst, pose_src, pose_dst, K_src, K_dst]
        unpacked_aux = [src_or_dst for aux in geom_aux for src_or_dst in transpose(aux)]
        # _, _, pose_src, pose_dst, K_src, K_dst = unpacked_aux
        # F_mat = kg.fundamental_from_projections(K_src @ invert_Rt(pose_src)[:, 0:3, 0:4], K_dst @ invert_Rt(pose_dst)[:, 0:3, 0:4])
        depth_src, depth_dst, pose_src, pose_dst, K_src, K_dst = unpacked_aux
        T0, T1 = invert_Rt(pose_src), invert_Rt(pose_dst)
        T_0to1 = torch.bmm((T1), invert_Rt(T0))[..., :4, :4]  # (4, 4)
        # T_1to0 = T_0to1.inverse()
        coord_src = [i[i.isfinite().all(dim=-1)] for i in coord_src]
        coord_dst = [i[i.isfinite().all(dim=-1)] for i in coord_dst]
        data = dict(
            mkpts0_f = coord_src,
            mkpts1_f = coord_dst,
            T_0to1 = T_0to1,
            K0 = K_src,
            K1 = K_dst,
        )
        compute_pose_errors(data)
        error = np.max(np.stack([data['R_errs'], data['t_errs']]), axis=0)
        # error = data['R_errs']
        error = torch.tensor(error, dtype=torch.float32)
        self.cur_metric(error)

    def epoch_end(self) -> None:
        mma_all = self.metric_weighted_mean()
        if mma_all is not None:
            self.module.log_dict({f"pose_auc/{th}": mma_all[i] for i, th in enumerate(self.thresh)}, sync_dist=True)
            self.log_stage_metric("", {f"pose_auc/{th}": mma_all[i] for i, th in enumerate(self.thresh)})
            # monitor metric for checkpointing and early termination
            self.log_monitor(mma_all[0])

            # log reprojection error distribution
            if self.logger is not None and isinstance(self.logger, TensorBoardLogger):
                error_all = torch.cat([metric.get_error_all().flatten() for metric in self.metrics])
                self.logger.experiment.add_histogram(self.format_stage_metric_name(f"log_pose_err_hist"),
                                                     error_all[error_all.isfinite()].log10(),
                                                     global_step=self.module.global_step)

            # monitor metric for checkpointing and early termination
            # self.module.log(f"monitor/{metric_prefix}_mma_{self.thresh[0]}px", mma_all[0])

