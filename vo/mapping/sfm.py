from dataclasses import dataclass
from itertools import chain
from multiprocessing import Pool
from typing import Sequence
import numpy as np
from vo.mapping.bundle_adjustment import BundleAdjuster
from vo.mapping.triangulation import Triangulator

from vo.pose_graph import PoseGraph, KeyFrame
from vo.utils import estimate_pose_PnP, estimate_pose_essential
from utils.geometry import denorm_thresh


@dataclass
class SFMInitializerConfig:
    min_n_corr = 50
    avg_parallax_norm = 0.025
    pnp_thresh_norm = 0.005
    ess_thresh_norm = 0.02
    ess_dist_thresh = 1000
    ba_iterations = 3


class SFMInitializer:
    def __init__(self, K: np.ndarray,
                 triangulator: Triangulator = None, bundle_adjuster: BundleAdjuster = None) -> None:
        self.K = K
        self.config = SFMInitializerConfig()
        self.triangulator = Triangulator(K=K) if triangulator is None else triangulator
        self.bundle_adjuster = BundleAdjuster(K=K, triangulator=self.triangulator) \
            if bundle_adjuster is None else bundle_adjuster

    def reconstruct(self, pose_graph: PoseGraph) -> bool:
        frames = pose_graph.keyframes
        # find good frame to pair with the last frame
        latest_frame = frames[-1]
        cand_frames = [frame for frame, n_corr in
                       sorted(latest_frame.n_corrs_by_frame, key=lambda x: x[1], reverse=True)
                       if n_corr > self.config.min_n_corr and frame.idx != latest_frame.idx]

        # try to find a frame that has both sufficient parallax and good pose estimation
        best_frame = None
        for frame in cand_frames:
            if self.check_parallax(frame, latest_frame, self.config.avg_parallax_norm):
                frame.pose = np.eye(4)
                success, pose = estimate_pose_essential(
                    latest_frame, frame, self.K, self.config.ess_thresh_norm, self.config.ess_dist_thresh)
                if success:
                    best_frame = frame
                    best_frame.register()
                    pose_graph.set_origin(best_frame)
                    latest_frame.pose = pose
                    latest_frame.register()
                    break

        if best_frame is None:
            # print('no best')
            return False

        if True: # origin: with Pool(8) as pool:
            pool=None
            # triangulate initial set of landmarks
            self.triangulator.triangulate_frame_features(latest_frame, in_place=True, thread_pool=pool)
            # temporarily reduce track length requirement since this is a 2-frame ba
            # self.bundle_adjuster.refine_local_bundle(pose_graph, frames=[best_frame, latest_frame], lm_min_track_len=2)

            # starting around best_frame, PnP and triangulate till both ends of the list
            best_iloc = best_frame.iloc
            frames_aft: Sequence[KeyFrame] = frames.iloc[best_iloc + 1: -1]
            frames_bfr: Sequence[KeyFrame] = frames.iloc[best_iloc - 1:: -1]
            last_successful_pose = best_frame.pose
            for frame in chain(frames_aft, [None], frames_bfr):
                if frame is None:
                    last_successful_pose = best_frame.pose
                    continue
                if frame.registered:
                    continue

                success = False
                # try to estimate wrt registered correspondent frames
                frame.pose = last_successful_pose
                for corr_frame in frame.correspondent_frames:
                    if not corr_frame.registered:
                        continue
                    success, pose = estimate_pose_essential(frame, corr_frame, self.K, self.config.ess_thresh_norm)
                    if success:
                        last_successful_pose = frame.pose = pose
                        break

                success, pose = estimate_pose_PnP(frame, self.K, self.config.pnp_thresh_norm)
                if success:
                    frame.register()
                    last_successful_pose = frame.pose = pose
                    self.triangulator.triangulate_frame_features(
                        frame, match_existing=True, in_place=True, thread_pool=pool)
                # else:
                #     print('pnp fail')

        errors = self.bundle_adjuster.refine_local_bundle(pose_graph, iteration=self.config.ba_iterations)

        # fail if final error is invalid
        return np.isfinite(errors[-1][1])

    def check_parallax(self, frame0: KeyFrame, frame1: KeyFrame, thresh_norm=0.05):
        disp = np.stack([fea0.xy - fea1.xy for fea0, fea1 in frame0.correspondences_with(frame1)])
        return np.linalg.norm(disp, axis=1).mean() > denorm_thresh(thresh_norm, self.K)
