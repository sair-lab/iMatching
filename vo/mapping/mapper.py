import collections
from dataclasses import dataclass
from enum import Enum
import sys
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from vo.mapping.bundle_adjustment import BundleAdjuster
from vo.mapping.sfm import SFMInitializer
from vo.mapping.triangulation import Triangulator
from vo.pose_graph import IndexedStore

from vo.tracking.motion_model import MotionModel

from ..pose_graph import Corrs, PoseGraph, Feature, KeyFrame, Landmark
from utils.geometry import reprojection_error


@dataclass
class MapConfig:
    stable_lm_min_track_len = 2   # keep as much as possible
    ba_iterations = 2


class Map:
    class State(Enum):
        UNINIT = -1
        OK = 0

    def __init__(self, K: np.ndarray, global_trans=np.eye(4), max_size=None) -> None:
        self.K = K
        self.pose_graph = PoseGraph()
        self.bundle_adjuster = BundleAdjuster(K=K) 
        self.trianglator = Triangulator(K=K)
        self.global_trans = global_trans
        self.config = MapConfig()
        self.state = Map.State.UNINIT
        self.max_size = sys.maxsize if max_size is None else max_size

    @property
    def n_frames(self):
        return len(self.pose_graph.keyframes)

    @property
    def n_stable_landmarks(self):
        return sum(1 for lm in self.pose_graph.landmarks
                   if lm.track_support_length >= self.config.stable_lm_min_track_len)

    def initialize(self):
        # maybe create a new BA for lower traj len limit
        initializer = SFMInitializer(self.K, self.trianglator, self.bundle_adjuster)
        return initializer.reconstruct(self.pose_graph)

    def transform_into(self, transf: np.ndarray):
        inv_transf = np.linalg.inv(transf)
        for frame in self.pose_graph.keyframes:
            frame.pose = inv_transf @ frame.pose
        for landmark in self.pose_graph.landmarks:
            landmark.position = inv_transf @ landmark.position

    def create_frame(
            self, pose: np.ndarray = None, fea_xy: Sequence[np.ndarray] = None, K: np.ndarray = None,
            fea_info: Sequence = None, frame_info: Any = None):
        return self.pose_graph.add_keyframe(pose=pose, fea_xy=fea_xy, K=K, fea_info=fea_info, frame_info=frame_info)

    def discard_extra_frames(self, n=None):
        # evacuate the oldest frames, starting with unregistered
        n = self.n_frames - self.max_size if n is None else n
        to_evacuate = [frame for i, frame in enumerate(self.pose_graph.keyframes)
                       if not frame.registered and i < n]

        n_to_evacuate = len(to_evacuate)
        if n_to_evacuate < n:
            to_evacuate_id = [frame.idx for frame in to_evacuate]
            frame_iter = iter(self.pose_graph.keyframes)
            for _ in range(n - n_to_evacuate):
                frame = next(frame_iter)
                while frame is not None and frame.idx in to_evacuate_id:
                    frame = next(frame_iter)
                to_evacuate.append(frame)

        move_origin = False
        for frame in to_evacuate:
            if self.pose_graph.is_origin(frame):
                move_origin = True
                self.pose_graph.set_origin(None)
            self.pose_graph.remove_keyframe(frame)
        if move_origin:
            self.pose_graph.set_origin(self.pose_graph.keyframes.iloc[0])

    def get_local_ref_frames(self, frame: KeyFrame, n: int) -> IndexedStore[KeyFrame, PoseGraph]:
        return self.pose_graph.keyframes.iloc[max(0, frame.iloc - n) : frame.iloc]

    def refine_local_bundle(self, frames: Sequence[KeyFrame] = None, keep_short_track=False):
        errors = self.bundle_adjuster.refine_local_bundle(self.pose_graph, frames=frames, iteration=self.config.ba_iterations, lm_min_track_len=(2 if keep_short_track else None))

        # fail if final error is invalid
        return np.isfinite(errors[-1][1]), errors
    
    def get_error(self, landmarks: Sequence[Landmark] = None, mean=True):
        if landmarks is None:
            landmarks = self.pose_graph.landmarks

        errs = []
        for lm in landmarks:
            for fea in lm.track:
                kf = fea.keyframe
                diff = reprojection_error(kf.pose[:3, :3], kf.pose[:3, 3], self.K, lm.position, fea.xy)
                errs.append(np.sqrt((diff ** 2).sum(axis=1)))

        if mean:
            return np.concatenate(errs).mean() if len(errs) > 0 else 0.
        else:
            return errs

    def get_output(self):
        if self.n_frames == 0:
            R, t = np.zeros((0, 3, 3)), np.zeros((0, 3, 1))
        else:
            R = np.stack([f.pose[:3, :3] for f in self.pose_graph.keyframes])
            t = np.stack([f.pose[:3, 3:4] for f in self.pose_graph.keyframes])
        landmarks = [lm for lm in self.pose_graph.landmarks if lm.track_length >= 3]
        if len(landmarks) == 0:
            point3d = np.zeros((0, 3))
        else:
            point3d = np.stack([lm.position for lm in landmarks])
        g_R, g_t = self.global_trans[None, :3, :3], self.global_trans[None, :3, 3:4]
        return g_R @ R, (g_R @ t + g_t)[:, :, 0], (g_R @ point3d[..., None] + g_t)[:, :, 0]

    def clear_mapping(self):
        for frame in self.pose_graph.keyframes:
            frame.pose = np.eye(4)

        for landmark in list(self.pose_graph.landmarks):
            self.pose_graph.remove_landmark(landmark)

    def reset(self):
        self.state = Map.State.UNINIT
        self.pose_graph = PoseGraph()
