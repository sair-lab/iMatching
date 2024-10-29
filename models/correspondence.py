import gc
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import kornia.geometry as kg
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from omegaconf import MISSING
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.tensorboard import SummaryWriter

from evaluation.evaluator.hpatches_matching import HPatchesMatchingEvaluator
from evaluation.evaluator.projection_based import EpipolarDistEvaluator, ReprojectionDistEvaluator
from evaluation.evaluator.reprojectionviz import TartanAirMatchVisualEvaluator
from evaluation.evaluator.sfm import SfMEvaluator
from utils.config import hydra_instantiable
from utils.geometry import (
    Projector,
    distort_denormalize,
    fundamental_matrix,
    invert_Rt,
    reprojection_error,
    undistort_normalize,
)
from utils.misc import dictseq2seqdict
from utils.visualization.multi import MultiVisualizer
from vo.pose_graph import KeyFrame
from vo.pose_graph.indexd_store import get_unique
from vo.pose_graph.pose_graph import Feature
from vo.tracking import Tracker

from .patch2pix_wrapper import Patch2PixWrapper
from .aspanformer_wrapper import ASpanFormerWrapper
from .dkm_wrapper import dkmWrapper
from .sift import Sift

class Criterion:
    def __call__(self, *args: Any, **kwds: Any) -> Tuple[torch.Tensor, ...]:
        pass


@dataclass
class ModelConfig:
    task: str = MISSING
    evaluator: Optional[str] = None
    criterion: Optional[Any] = None
    reproj_thresh: List[float] = field(default_factory=lambda: [1., 2., 5., 10.])


@hydra_instantiable(name="model", group="model")
class CorrespondenceModel(pl.LightningModule):
    """
    When keypoint is being set to optional, the matching module directly takes the image as input and produce a matching. 
    """
    def __init__(self, matching, keypoint: Optional[nn.Module]=None, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.keypoint_model = keypoint
        self.matching_model = matching

        self.vo = Tracker(self.keypoint_model, self.matching_model)
        self.chunk_size = 2

        self.caps_crit: Criterion = cfg.criterion

        self.viz_2d = None
        self.tensorboard: SummaryWriter = None
        self.last_loss = 0.
        self.min_lm_track_length = -1

        # validation
        if cfg.evaluator == "sfm_inlier_rate":
            self.evaluator = SfMEvaluator(self, self.vo)
        elif cfg.evaluator == "reproj_mma":
            self.evaluator = ReprojectionDistEvaluator(self, cfg.reproj_thresh)
        elif cfg.evaluator == "epipolar_mma":
            self.evaluator = EpipolarDistEvaluator(self, cfg.reproj_thresh)
        elif cfg.evaluator == "pose_auc":
            from evaluation.evaluator.pose_based import PoseEvaluator
            self.evaluator = PoseEvaluator(self, cfg.reproj_thresh)
        # elif cfg.task == "test-hpatches":
        #     self.evaluator = HPatchesMatchingEvaluator(self, cfg.reproj_thresh)
        # elif cfg.task == "test-localization-7scenes":
        #     self.evaluator = LocalizationEvaluator(self, None, thresh=cfg.reproj_thresh)
        # elif cfg.task in ("visualize-tartanair-matches", "visualize-7scenes-matches"):
        #     self.evaluator = TartanAirMatchVisualEvaluator(self, 1.)
    def to(self, device):
        if type(self.matching_model) is Patch2PixWrapper:
            self.matching_model.model.device = device
        return super().to(device)

    def on_train_start(self):
        self.matching_model.device = self.device
        if self.logger is not None:
            self.tensorboard = self.logger.experiment
            self.viz_2d = MultiVisualizer(display="tensorboard", writer=self.tensorboard)

    def training_step(self, batch, batch_idx):
        (image, depth, pose, K), (series_key, idx) = batch
        loss, batch_good = 0, False
        for image_, depth_, pose_, K_, key_, idx_ in zip(image, depth, pose, K, zip(*series_key), idx):
            frame_locator = dictseq2seqdict({'series': zip(*key_), 'idx': idx_.tolist()})
            # just to be safe
            try:
                loss_seq = self._step_single_sequence(image_, depth_, pose_, K_, frame_locator, idx_.tolist(), batch_idx)
            except Exception as e:
                traceback.print_exc()
                self.vo.reset()
                gc.collect()
                loss_seq = None

            if loss_seq is not None:
                loss = loss + loss_seq
                batch_good = True
                self.last_loss = loss.item() if type(loss) is torch.Tensor else loss

        return loss if batch_good else torch.tensor(
            self.last_loss, dtype=torch.float32, device=self.device, requires_grad=True)

    def on_train_epoch_end(self):
        if hasattr(pl.utilities, "finite_checks"):
            pl.utilities.finite_checks.detect_nan_parameters(self)
        else:
            torch.autograd.set_detect_anomaly(True, check_nan=True)

    def _step_single_sequence(self, imgs, depths, poses, Ks, frame_info=None, timestamps=None, batch_idx=None):
        self.vo.reset()
        gc.collect()

        self.vo.track(imgs, Ks, frame_info, timestamps=timestamps, slide_window=False)

        new_frames = self.vo.map.pose_graph.keyframes

        use_gt = False
        if use_gt:
            self.inject_gt(imgs, depths, poses, Ks, new_frames)

        ba_success, err_all = self.vo.map.refine_local_bundle()
        if self.logger is not None:
            for i, err in enumerate(err_all):
                self.log_dict({f"ba/{i}/initial": err[0], f"ba/{i}/final": err[1]})

        loss = 0

        # calculate reprojection error perframe
        shorter_edge, longer_edge = min(imgs.shape[-2:]), max(imgs.shape[-2:])
        errors_c, errors_f, errors_cycle_c, errors_cycle_f, stds_c, stds_f, mean_std_c, mean_std_f, n_inliers = self.reprojection_loss(new_frames, shorter_edge, Ks)
        loss += 1 * (errors_c + errors_f).mean() / longer_edge
        # loss += 0.1 * (errors_cycle_c + errors_cycle_f).mean() / longer_edge
        # loss += 1e-1 * (stds_c.log() + stds_c.log()).mean()

        self.log_dict({'ba_losses/mean_reprojection/coarse': errors_c.mean(), 'ba_losses/mean_reprojection/fine': errors_f.mean()})
        # self.log("ba_losses/cycle_consistency", {'coarse': errors_cycle_c.mean(), 'fine': errors_cycle_f.mean()})
        self.log_dict({'ba_losses/mean_std/coarse': mean_std_c, 'ba_losses/mean_std/fine': mean_std_f})
        self.log("ba_losses/n_points", float(n_inliers))

        if 'caps.CAPS' in str(type(self.matching_model)):
            eloss, eloss_c, eloss_f, closs_c, closs_f, std_loss = self.epipolar_loss(new_frames, Ks.cpu().numpy(), imgs.shape[-2:])
            loss += eloss

            self.log('caps_losses/total', eloss)
            self.log_dict({'caps_losses/epipolar/coarse': eloss_c, 'caps_losses/epipolar/fine': eloss_f})
            self.log_dict({'caps_losses/cycle_consistency/coarse': closs_c, 'caps_losses/cycle_consistency/fine': closs_f})
            self.log('caps_losses/std', std_loss)
        elif type(self.matching_model) is Patch2PixWrapper:
            pass
            # eloss = self.p2p_loss_wrapper(new_frames, Ks.cpu().numpy(), imgs.shape[-2:])
            # loss += eloss
            # self.log('p2p_losses/total', eloss)
        elif 'ASpanFormerWrapper' in str(type(self.matching_model)):
            pass
            # loss += self.aspan_loss_epi(new_frames, Ks.cpu().numpy(), imgs.shape[-2:])
        elif 'dkm' in str(type(self.matching_model)):
            pass
        else:
            raise NotImplementedError

        if self.global_step % 50 == 0 and self.local_rank == 0:
            # self.log_pose_graph(self.global_step, self.vo.map.pose_graph)

            points0_, points1_, colors, sizes = [], [], [], []
            for frame0, frame1 in zip(new_frames.iloc[:-1], new_frames.iloc[1:]):
                corrs = frame0.correspondences_with(frame1)
                fea0_xy, fea1_xy = [], []
                for fea0, fea1 in corrs:
                    fea0_xy.append(torch.from_numpy(fea0.xy))
                    fea1_xy.append(torch.from_numpy(fea1.xy))
                    colors.append(0 if fea0.has_landmark and fea1.has_landmark and fea0.landmark.idx == fea1.landmark.idx else 1)
                fea0_xy = torch.stack(fea0_xy) if len(fea0_xy) > 0 else torch.zeros((0, 2))
                fea1_xy = torch.stack(fea1_xy) if len(fea1_xy) > 0 else torch.zeros((0, 2))
                points0_.append(fea0_xy)
                points1_.append(fea1_xy)
                sizes.append(len(corrs))
            padded_size = max(sizes)
            for i, (pts0, pts1) in enumerate(zip(points0_, points1_)):
                points0_[i] = torch.full((padded_size, 2), np.nan)
                points0_[i][:len(pts0)] = pts0
                points1_[i] = torch.full((padded_size, 2), np.nan)
                points1_[i][:len(pts1)] = pts1

            if self.viz_2d is not None:
                points0 = distort_denormalize(torch.stack(points0_), Ks[:-1])
                points1 = distort_denormalize(torch.stack(points1_), Ks[1:])
                self.viz_2d.showmatch(imgs[:-1], points0, imgs[1:], points1, None, sizes, color="bwr", values=colors, pts_colors=('white', 'white'), vlim=[0, 1], step=self.global_step, alpha=0.5)
        
        self.vo.reset()
        gc.collect()

        if not ba_success or not torch.isfinite(loss):
            # skip if ba failed or nan loss
            return self.dummy_loss()
        else:
            return loss

    def on_validation_epoch_start(self):
        self.evaluator.on_start()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.evaluator.step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        self.evaluator.epoch_end()
        self.evaluator.reset()

    def on_test_start(self):
        self.evaluator.on_start()

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.evaluator.step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self) -> None:
        self.evaluator.epoch_end()
        self.evaluator.reset()

    def inject_gt(self, imgs, depths, poses_gt, Ks, new_frames: Iterable[KeyFrame]):
        def is_detected_feature(fea: Feature):
            return fea.info is not None and 'fea_color' in fea.info
        n_fea_per_frame = self.keypoint_model.max_n_fea
        projector = Projector()
        fea_xys = np.full((self.vo.map.n_frames, n_fea_per_frame, 2), np.nan, dtype=np.float32)

        for i, kf in enumerate(new_frames):
            for j, fea in enumerate(kf.features):
                    # care only about detected features
                if is_detected_feature(fea):
                    fea_xys[i, j] = fea.xy
        poses_gt_r = invert_Rt(poses_gt[0:1]) @ poses_gt
        fea_xys = torch.from_numpy(fea_xys).to(self.device, torch.float32)
        fea_world = projector.pix2world(undistort_normalize(fea_xys, Ks), depths, invert_Rt(poses_gt_r), Ks).cpu().numpy()

        for i, kf in enumerate(new_frames):
            for j, fea in enumerate(kf.features):
                    # care only about detected features
                if fea.info is None and fea.has_landmark:
                    fea.landmark.position = fea_world[i, j]
            kf.pose = poses_gt_r[i].cpu().numpy()

    def reprojection_loss(self, keyframes: Sequence[KeyFrame], shorter_edge, Ks):
        dummy = torch.tensor([], device=self.device, requires_grad=True)
        errors_c_, stds_c_, errors_f_, stds_f_, errors_cycle_c_, errors_cycle_f_ = [dummy], [dummy], [dummy], [dummy], [dummy], [dummy]
        new_landmarks = []
        for kf in keyframes:
            p = []
            output = {}
            for landmark in kf.landmarks:
                if landmark.track_length < self.min_lm_track_length:
                    continue
                new_landmarks.append(landmark)
                for fea in landmark.track:
                    if fea.info is not None and fea.feature_id[0] == kf.idx and 'xy' in fea.info:
                        for k, v in fea.info.items():
                            output.setdefault(k, []).append(v)
                        p.append(torch.from_numpy(landmark.position.astype(np.float32)).to(self.device))
            kf_R, kf_t = torch.from_numpy(kf.R).to(self.device, torch.float), torch.from_numpy(kf.t).to(self.device, torch.float)
            kf_K = torch.from_numpy(kf.K).to(self.device, torch.float)
            dist_coeff = self.vo.dist_coeff
            if len(p) > 0:
                # reprojection loss, taking distortion into account
                coord2_ec_u = undistort_normalize(torch.stack(output['coord2_ec']), kf_K, dist_coeff, normalize=False)
                coord2_ef_u = undistort_normalize(torch.stack(output['xy']), kf_K, dist_coeff, normalize=False)
                errors_c_.append(reprojection_error(kf_R, kf_t, kf_K, torch.stack(p), coord2_ec_u).norm(dim=-1))
                errors_f_.append(reprojection_error(kf_R, kf_t, kf_K, torch.stack(p), coord2_ef_u).norm(dim=-1))

                # cycle consistency loss
                # errors_cycle_c_.append((torch.stack(output['coord1_lc']) - torch.stack(output['xy'])).norm(dim=-1))
                # errors_cycle_f_.append((torch.stack(output['coord1_lf']) - torch.stack(output['xy'])).norm(dim=-1))

                # std loss
                # stds_c_.append(torch.stack(output['std_c']))
                # stds_f_.append(torch.stack(output['std_f']))

        errors_c = torch.clamp(torch.cat(errors_c_), max=shorter_edge * 0.5)
        errors_f = torch.clamp(torch.cat(errors_f_), max=shorter_edge * 0.5)
        errors_cycle_c = torch.clamp(torch.cat(errors_cycle_c_), max=shorter_edge * 0.25)
        errors_cycle_f = torch.clamp(torch.cat(errors_cycle_f_), max=shorter_edge * 0.25)
        stds_c = torch.cat(stds_c_)
        stds_f = torch.cat(stds_f_)
        mean_std_c = stds_c[stds_c.isfinite()].mean()
        mean_std_f = stds_f[stds_f.isfinite()].mean()
        n_inliers = len(list(get_unique(new_landmarks)))
        return errors_c, errors_f, errors_cycle_c, errors_cycle_f, stds_c, stds_f, mean_std_c, mean_std_f, n_inliers

    def epipolar_loss(self, keyframes: Sequence[KeyFrame], Ks: np.ndarray, img_size: Tuple[int, int]):
        ctof_keys = ['coord2_ec', 'coord2_ef', 'coord1_lc', 'coord1_lf', 'std_c', 'std_f', 'std_lc', 'std_lf']
        coord1_, data_, fmatrix_ = [], {k: [] for k in ctof_keys}, []
        for kf, K in zip(keyframes, Ks):
            for ref_frame_idx, corr_info in kf.info['corr_info'].items():
                ref_frame: KeyFrame = self.vo.map.pose_graph.keyframes[ref_frame_idx]
                coord1_.append(ref_frame.info['xy'])
                for k, v in corr_info.items():
                    data_[k].append(v)
                fmatrix_.append(torch.from_numpy(fundamental_matrix(ref_frame.pose, K, kf.pose, K)).to(coord1_[0]))

        coord1 = torch.stack(coord1_)
        data = {k: torch.stack(v) for k, v in data_.items()}

        return self.caps_crit(coord1, data, torch.stack(fmatrix_), None, img_size)

    def dummy_loss(self):
        dummy_loss = 0
        model = self.matching_model
        for param in model.parameters():
            dummy_loss = dummy_loss + param.sum() * 0
        return dummy_loss

    def aspan_loss_epi(self, keyframes: Sequence[KeyFrame], Ks: np.ndarray, img_size: Tuple[int, int]):
        data_keys = ['coord1_ef', 'coord2_ef']
        data_, fmatrix_ = {k: [] for k in data_keys}, []
        for kf, K in zip(keyframes, Ks):
            for ref_frame_idx, corr_info in kf.info['corr_info'].items():
                ref_frame: KeyFrame = self.vo.map.pose_graph.keyframes[ref_frame_idx]
                for k in data_keys:
                    data_[k].append(corr_info[k])
                fmatrix_.append(torch.from_numpy(fundamental_matrix(ref_frame.pose, K, kf.pose, K)).to(data_[data_keys[1]][0]))

        data = {k: torch.stack(v) for k, v in data_.items()}

        from ext.patch2pix.networks.utils import sampson_dist
        pairs = torch.cat([data['coord1_ef'], data['coord2_ef']], dim=-1)
        loss = 0
        for pair, f in zip(pairs, fmatrix_):
            epi_loss = sampson_dist(pair, f)
            loss += epi_loss[~epi_loss.isnan()].sum()
        return loss

        def homogenize(coord):
            coord = torch.cat((coord, torch.ones_like(coord[:, :, [0]])), -1)
            return coord
        def epipolar_cost(coord1, coord2, fmatrix):
            coord1_h = homogenize(coord1).transpose(1, 2)
            coord2_h = homogenize(coord2).transpose(1, 2)
            epipolar_line = fmatrix.bmm(coord1_h)  # Bx3xn
            epipolar_line_ = epipolar_line / torch.clamp(torch.norm(epipolar_line[:, :2, :], dim=1, keepdim=True), min=1e-8)
            essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_, dim=1))  # Bxn
            return essential_cost        

        # second
        a = epipolar_cost(data['coord1_ef'], data['coord2_ef'], torch.stack(fmatrix_))

        return a[~a.isnan()].mean()

    def p2p_loss_wrapper(self, keyframes: Sequence[KeyFrame], Ks: np.ndarray, img_size: Tuple[int, int]):
        data_keys = ['coarse_matches', 'mid_matches', 'fine_matches', 'mid_probs', 'fine_probs']
        data_, fmatrix_ = {k: [] for k in data_keys}, []
        for kf, K in zip(keyframes, Ks):
            for ref_frame_idx, corr_info in kf.info['corr_info'].items():
                ref_frame: KeyFrame = self.vo.map.pose_graph.keyframes[ref_frame_idx]
                for k in data_keys:
                    data_[k].append(corr_info[k])
                fmatrix_.append(torch.from_numpy(fundamental_matrix(ref_frame.pose, K, kf.pose, K)).to(data_[data_keys[1]][0]))

        data = {k: torch.stack(v) for k, v in data_.items()}

        return self.matching_model.loss(fmatrix_, **data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.matching_model.parameters(), lr=1e-4)

    @rank_zero_only
    def log_pose_graph(self, step, graph, prefix=None):
        if self.tensorboard is None:
            return
        if prefix is None:
            prefix = self.tensorboard.log_dir
        # save pose_graph for visualization
        path = Path(prefix) / 'pose_graph' / f'{step}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, path)
