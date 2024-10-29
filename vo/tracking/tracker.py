from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence
import cv2
import os

import kornia.geometry as kg
import numpy as np
import scipy.spatial as ss
import torch

from utils.geometry import coord_list_grid_sample, distort_denormalize, undistort_normalize
from utils.misc import dictseq2seqdict, sized_chunk
from vo.mapping.mapper import Map
from vo.pose_graph import KeyFrame
from vo.tracking.motion_model import MotionModel

from ..utils import estimate_pose_essential, estimate_pose_PnP
from pathlib import Path

@dataclass
class TrackerConfig:
    window_size = 3
    n_ref_init = 2
    n_ref_tracking = 2
    min_n_tracked_lm = 50
    pnp_thresh_norm = 0.015
    ess_thresh_norm = 0.015
    ess_dist_thresh = 1000
    fea_merge_thresh = 0.01  # the thres for kdtree merging

    def __post_init__(self):
        if self.window_size < self.n_ref_init:
            raise ValueError('Cannot use more frames than existing in local map as ref frames.')


class Tracker():
    class State(Enum):
        UNINIT = -1
        OK = 0
        LOST = -2

    def __init__(
            self, extractor, matcher, frame_rate=1., K: Optional[np.ndarray] = None, normalize_coord: bool = True,
            dist_coeff: Optional[np.ndarray] = None) -> None:
        self._matcher = matcher
        self._extractor = extractor
        self.config = TrackerConfig()
        self.dt = 1. / frame_rate
        self.motion_model = MotionModel()
        # K for internal processing
        self.K = np.eye(3, dtype=np.float32)
        # K of input image
        self.K_img = K
        self.normalize_coord = normalize_coord
        if not normalize_coord:
            if self.K_img is not None:
                # use image K for internal processing
                self.K = self.K_img
            else:
                print("K is not provided during initialization of Tracker, "
                "and normalize_coord is not enabled. It must be provided during tracking when image is given.")
        self.dist_coeff = torch.from_numpy(dist_coeff).to(torch.float) if dist_coeff is not None else None
        self.map = Map(K=self.K, max_size=self.config.window_size)
        self.n_frames_observed = 0
        self.state = Tracker.State.UNINIT
        self.debug = True
        self.last_gba = 0
        self.last_keyframe: KeyFrame = None
    
    def track(
            self, images: torch.Tensor, K: Optional[torch.Tensor] = None, image_info: Optional[Iterable] = None,
            timestamps: Sequence[float] = None, slide_window=True):
        B, _, H, W = images.shape

        if timestamps is None:
            timestamps = (np.arange(B) + self.n_frames_observed) * self.dt
        self.n_frames_observed += B

        # resolution order: comes with image, set at init, infered from image dimensions
        if K is None:
            if self.K_img is not None:
                K = torch.from_numpy(self.K_img).to(torch.float)
            else:
                fx, fy, cx, cy = W // 2, W // 2, W // 2, H // 2
                K = torch.tensor([[fx, 0., cx],
                                  [0., fy, cy],
                                  [0., 0., 1.]])
            K = K[None].expand(B, 3, 3).to(images)


        new_frames = self.match_frames(images, K, image_info)

        if self.state == Tracker.State.UNINIT:
            # estimate initial map after having enough frames
            if self.map.n_frames >= self.config.window_size:
                init_success = self.map.initialize()
                # complete init if enough number of landmarks
                if init_success and self.map.n_stable_landmarks > self.config.min_n_tracked_lm:
                    for frame, timestamp in zip(new_frames, timestamps):
                        if frame.registered:
                            self.motion_model.update(frame.pose, timestamp)
                            self.last_keyframe = frame
                    self.state = Tracker.State.OK
                else:
                    # remove landmarks and reset poses
                    self.map.clear_mapping()
        elif self.state == Tracker.State.OK:
            for frame, timestamp in zip(new_frames, timestamps):
                success = self.track_frame(frame, timestamp)
                if success:
                    self.map.refine_local_bundle(frames=[frame], keep_short_track=True)
                else:
                    self.state = Tracker.State.LOST

            # global ba
            if new_frames[-1].idx - self.last_gba >= 10:
                self.map.refine_local_bundle()
                self.last_gba = new_frames[-1].idx

        if slide_window:
            self.map.discard_extra_frames()

    def add_frames(self, desc, xy, n_fea=None, K=None, frame_info=None, fea_info=None):
        n_fea = [np.isfinite(c).all(axis=1).sum() for c in xy] if n_fea is None else n_fea
        fri = [dict() for _ in range(len(desc))] if frame_info is None else frame_info
        fei = [None] * len(desc) if fea_info is None else fea_info
        new_frames: List[KeyFrame] = []

        for d, c, n, cam, fr, fe in zip(desc, xy, n_fea, K, fri, fei):
            # create new frame and save metadata without adding correspondences
            fr = fr.copy()
            fr.update({"desc": d, "xy": c, "n_fea": n})
            if c is not None:
                c_n = undistort_normalize(c, cam, self.dist_coeff, self.normalize_coord)
                c_n = c_n[:n].detach().cpu().numpy()
            else:
                c_n = None
            new_frames.append(self.map.create_frame(
                pose=np.eye(4), fea_xy=c_n,
                K=cam.detach().cpu().numpy(), frame_info=fr, fea_info=fe))
        return new_frames

    def match_frames(self, images, K, image_info=None):
        B, _, H, W = images.shape
        image_info = [None] * B if image_info is None else image_info
        frame_info = dictseq2seqdict({'img': images, 'img_info': image_info})
        max_ref = self.config.n_ref_init if self.state == Tracker.State.UNINIT else self.config.n_ref_tracking
        fea_info = None

        if self._extractor is not None:
            coords, desc, n_points = self._extractor(images)
            # for visualization
            if self.debug:
                colors = coord_list_grid_sample(images, undistort_normalize(coords, K, self.dist_coeff)).detach().cpu().numpy()
                fea_info = [dictseq2seqdict({'fea_color': colors_[:n]}) for colors_, n in zip(colors, n_points)]
            # use past <window_size> frames as reference
            new_frames: Iterable[KeyFrame] = self.add_frames(desc, coords, n_points, K, frame_info, fea_info)
            ref_frames: Iterable[Iterable[KeyFrame]] = [self.map.get_local_ref_frames(frame, max_ref) for frame in new_frames]

            xy_r, n_r, aux_r, aux_n, n_ref = [], [], [], [], []
            # each frame in window vs this frame
            for window, nf in zip(ref_frames, new_frames):
                for rf in window:
                    xy_r.append(rf.info['xy'])
                    n_r.append(rf.info['n_fea'])
                    aux_r.append(rf.info['img'])
                    aux_n.append(nf.info['img'])
                n_ref.append(len(window))
            xy_r = torch.stack(xy_r)
            n_r = torch.stack(n_r)
            aux_r = torch.stack(aux_r)
            aux_n = torch.stack(aux_n)

            if len(n_r) < 1:
                return

            corrs_idx_all, corr_xy_all, corr_info_all = self._matcher(xy_r, n_r, aux_r, aux_n)
        else:
            new_frames: Iterable[KeyFrame] = self.add_frames([None]*B, [None]*B, [None]*B, K, frame_info, fea_info)
            ref_frames: Iterable[Iterable[KeyFrame]] = [self.map.get_local_ref_frames(frame, max_ref) for frame in new_frames]
            aux_r, aux_n, n_ref = [], [], []
            # each frame in window vs this frame
            for window, nf in zip(ref_frames, new_frames):
                for rf in window:
                    aux_r.append(rf.info['img'])
                    aux_n.append(nf.info['img'])
                n_ref.append(len(window))
            aux_r = torch.stack(aux_r)
            aux_n = torch.stack(aux_n)
            corrs_idx_all, corr_xy_all, n_r, corr_info_all = self._matcher(aux_r, aux_n)

        # split corrs by number of ref frames matching with that frame
        # zip(new_frames, ref_frames) to make type hint happy
        # (zip type hint doesn't work for >5 inputs?)
        for (frame, refs), corrs_idx, corr_xy, corr_num, corr_info in zip(zip(new_frames, ref_frames),
                                                                        sized_chunk(corrs_idx_all, n_ref),
                                                                        sized_chunk(corr_xy_all, n_ref),
                                                                        sized_chunk(n_r, n_ref),
                                                                        sized_chunk(corr_info_all, n_ref)):
            frame.info['corr_info'] = {ref_frame.idx: info for ref_frame, info in zip(refs, corr_info)}
            for ref, corr, fea_from_ref, n_from_ref, info in zip(refs, corrs_idx, corr_xy, corr_num, corr_info):
                if self._extractor is None:
                    # r, n
                    fea_r = fea_from_ref[:n_from_ref, :2]
                    fea_from_ref = fea_from_ref[..., 2:]
                fea_from_ref, corr = fea_from_ref[:n_from_ref], corr[:n_from_ref]
                info = {k: v[:n_from_ref] for k, v in info.items()}
                n_existing_fea = len(frame.features)
                r_existing_fea = len(ref.features)
                # create features from matching with ref frame
                fea_info_ = {'xy': fea_from_ref}
                fea_info_.update(info)
                for k in fea_info_:
                    assert len(fea_info_[k]) == len(fea_info_['xy'])
                new_fea_xy = undistort_normalize(
                    fea_from_ref.detach(),
                    torch.from_numpy(frame.K).to(torch.float).to(fea_from_ref),
                    self.dist_coeff, self.normalize_coord).cpu().numpy()
                frame.add_features(new_fea_xy, info=dictseq2seqdict(fea_info_))

                # shift indices as new features are added
                corr = corr.detach().cpu().numpy()[..., [1, 0]]
                corr[:, 0] += n_existing_fea

                if self._extractor is None:
                    fea_info_ = {'xy': fea_r, 'coord2_ec': info['coord1_ec'], 'coord2_ef': info['coord1_ef']}

                    ref_fea_xy = undistort_normalize(
                        fea_r.detach(),
                        torch.from_numpy(ref.K).to(torch.float).to(fea_r),
                        self.dist_coeff, self.normalize_coord).cpu().numpy()
                    ref.add_features(ref_fea_xy, info=dictseq2seqdict(fea_info_)) 
                    corr[:, 1] += r_existing_fea

                frame.add_correspondences(ref, corr)
                # merge new features with old features if close
                # by linking features extracted from frame to those matched from ref frame
                if True:
                    if n_existing_fea > 0:
                        old_fea_xy = np.stack([fea.xy for idx, fea in enumerate(frame.features) if idx < n_existing_fea])
                    else:
                        old_fea_xy = np.zeros((0, 2), dtype=np.float32)
                    old_fea_xy_dist = distort_denormalize(
                        torch.from_numpy(old_fea_xy),
                        torch.from_numpy(frame.K),
                        self.dist_coeff, self.normalize_coord).cpu().numpy()
                    new_fea_xy_dist = fea_from_ref.detach().cpu().numpy()
                    kd_old, kd_new = ss.KDTree(old_fea_xy_dist), ss.KDTree(new_fea_xy_dist)
                    self_corr = [(i, js[0] + n_existing_fea)
                                for i, js in enumerate(kd_old.query_ball_tree(kd_new, r=self.config.fea_merge_thresh))
                                if len(js) > 0]
                    frame.add_correspondences(frame, self_corr)
                if self._extractor is None:
                    if r_existing_fea > 0:
                        old_fea_xy = np.stack([fea.xy for i, fea in enumerate(ref.features) if i < r_existing_fea])  #  
                    else:
                        old_fea_xy = np.zeros((0, 2), dtype=np.float32)
                    old_fea_xy_dist = distort_denormalize(
                        torch.from_numpy(old_fea_xy),
                        torch.from_numpy(ref.K),
                        self.dist_coeff, self.normalize_coord).cpu().numpy()
                    new_fea_xy_dist = fea_r.detach().cpu().numpy()
                    kd_old, kd_new = ss.KDTree(old_fea_xy_dist), ss.KDTree(new_fea_xy_dist)
                    self_corr = [(i, js[0] + r_existing_fea)
                                for i, js in enumerate(kd_old.query_ball_tree(kd_new, r=self.config.fea_merge_thresh))
                                if len(js) > 0]
                    ref.add_correspondences(ref, self_corr)                    
        return new_frames

    def track_frame(self, frame: KeyFrame, timestamp):
        # try to initialize by registrating existing landmarks or estimating homography
        if self.motion_model.ready:
            frame.pose = self.motion_model.predict(timestamp)
        else:
            frame.pose = self.last_keyframe.pose

        success, new_pose = estimate_pose_PnP(frame, self.K, self.config.pnp_thresh_norm)
        if not success:
            success, new_pose = estimate_pose_essential(
                frame, self.last_keyframe,
                self.K, self.config.ess_thresh_norm,
                self.config.ess_dist_thresh)
            if success:
                frame.pose = new_pose
                success_, new_pose_ = estimate_pose_PnP(frame, self.K, self.config.pnp_thresh_norm)
                if success_:
                    frame.pose = new_pose_
                    success = True

        if success:
            frame.pose = new_pose
            frame.register()
            self.motion_model.update(new_pose, timestamp)
            self.map.trianglator.triangulate_frame_features(frame, match_existing=True, in_place=True)
            self.last_keyframe = frame

        return success

    def reset(self):
        self.n_frames_observed = 0
        self.last_gba = 0
        self.map.reset()
        self.motion_model = MotionModel()
        self.state = Tracker.State.UNINIT
