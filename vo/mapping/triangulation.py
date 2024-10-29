from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple
import numpy as np

from vo.pose_graph import Feature, KeyFrame, Landmark
from utils.geometry import denorm_thresh, reprojection_error, triangulate_multiview, triangulate_ransac


@dataclass
class TriangulatorConfig:
    ransac_thresh_norm = 0.005
    reproj_thresh_norm = 0.005
    max_iter = 100


class Triangulator:
    def __init__(self, K: np.ndarray) -> None:
        self.K = K
        self.config = TriangulatorConfig()
        self.ransac_thresh = denorm_thresh(self.config.ransac_thresh_norm, K)
        self.reproj_thresh = denorm_thresh(self.config.reproj_thresh_norm, K)

    def triangulate_frame_features(
            self, frame: KeyFrame, match_existing=False, in_place=False, thread_pool: Pool = None):
        if not frame.registered:
            raise ValueError('Frame must be registered before triangulation.')

        if match_existing:
            for feature in frame.correspondent_features:
                # grow track if possible
                if not feature.has_landmark:
                    for corr_fea in feature.correspondences:
                        if corr_fea.has_landmark and self.check_reprojection(feature, corr_fea.landmark):
                            corr_fea.landmark.add_feature(feature)
                            break

        poses, xys, tracks = [], [], []
        feature_id_in_use = set()
        for feature in frame.correspondent_features_proper:
            if feature.has_landmark or feature.feature_id in feature_id_in_use:
                continue

            # collate feature info
            poses_, xys_, track_ = [feature.keyframe.pose], [feature.xy], [feature]
            frame_ids = {feature.frame_id}
            for ref_fea in feature.get_connected_features(5)[1:]:
                ref_frame = ref_fea.keyframe
                # NB1: no need to triangulate w.r.t. features with landmark (should add to landmark directly)
                # NB2: a feature from one frame can be connected to multiple features on the other frame
                # through same-frame-correspondences (feature mergence). In this case, only one feature from
                # the other frame is chosen
                if not ref_fea.has_landmark and ref_frame.registered and \
                        ref_frame.idx not in frame_ids and ref_fea.feature_id not in feature_id_in_use:
                    poses_.append(ref_frame.pose)
                    xys_.append(ref_fea.xy)
                    track_.append(ref_fea)
                    frame_ids.add(ref_frame.idx)

            if len(poses_) > 1:
                poses.append(poses_)
                xys.append(xys_)
                tracks.append(track_)
                feature_id_in_use.update(fea.feature_id for fea in track_)

        kernel = partial(triangulate_kernel, K=self.K, ransac_thresh=self.ransac_thresh, max_iter=self.config.max_iter)
        if thread_pool is not None:
            results = thread_pool.starmap(kernel, zip(poses, xys))
        else:
            results = [kernel(poses_, xys_) for poses_, xys_ in zip(poses, xys)]

        # create landmark for each successful triangulation
        positions_ret: List[np.ndarray] = []
        tracks_ret: List[List[Feature]] = []
        for (position, inlier), track in zip(results, tracks):
            filtered_track = [fea for fea, inlier_ in zip(track, inlier) if inlier_]
            if len(filtered_track) >= 2:
                positions_ret.append(position)
                tracks_ret.append(filtered_track)

        if in_place:
            pose_graph = frame.owner
            for position, track in zip(positions_ret, tracks_ret):
                pose_graph.add_landmark(position, track)

        return positions_ret, tracks_ret

    def check_reprojection(self, feature: Feature, landmark: Landmark, thresh=None):
        frame = feature.keyframe
        error, z = reprojection_error(
            frame.R, frame.t, self.K, landmark.position[None],
            feature.xy[None],
            return_z=True)
        thresh = self.reproj_thresh if thresh is None else thresh
        return (error[0] ** 2).sum() < thresh ** 2 and z[0] > 0


def triangulate_kernel(poses: np.ndarray, xys: np.ndarray, K: np.ndarray,
                       ransac_thresh=1, max_iter=100) -> Tuple[np.ndarray, np.ndarray]:
    if len(poses) <= 1:
        return np.full(3, np.nan), np.zeros(len(poses), dtype=bool)
    elif len(poses) == 2:
        position, inliers = triangulate_multiview(np.stack(poses), np.stack(xys), K)
    else:
        position, inliers = triangulate_ransac(np.stack(poses), np.stack(xys), K,
                                               thresh=ransac_thresh, max_trials=min(2 ** (len(poses) + 1), max_iter))
    return position, inliers
