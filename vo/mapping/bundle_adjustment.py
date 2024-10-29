import collections
from dataclasses import dataclass
from itertools import chain
import gtsam
import gtsam.symbol_shorthand as gtss
import gtsam.noiseModel as noiseModel
import numpy as np

from typing import List, Optional, Sequence, Tuple

from vo.mapping.triangulation import Triangulator

from ..pose_graph import KeyFrame, Landmark, PoseGraph, get_unique
from utils.geometry import reprojection_error, denorm_thresh


@dataclass
class BundleAdjusterConfig:
    lm_merge_reproj_thresh_norm = 0.00625
    lm_max_cos_parallax = 0.9998
    lm_min_track_len = 2   # keep as much as possible
    ba_unstable_tracks = True


class BundleAdjuster:
    def __init__(self, K: np.ndarray, triangulator: Triangulator = None) -> None:
        self.K = K
        self.config = BundleAdjusterConfig()
        self.lm_merge_reproj_thresh = denorm_thresh(self.config.lm_merge_reproj_thresh_norm, K)
        self.triangulator = Triangulator(K=K) if triangulator is None else triangulator

    def refine_local_bundle(self, pose_graph: PoseGraph, frames: Sequence[KeyFrame] = None, iteration: int = 3, lm_min_track_len: Optional[int] = None):
        if frames is None:
            frames = [frame for frame in pose_graph.keyframes if frame.registered]

        lm_min_track_len = self.config.lm_min_track_len if lm_min_track_len is None else lm_min_track_len

        errors: List[Tuple[float, float]] = []
        for _ in range(iteration):
            local_bundle = list(get_unique(
                [frame for nf in frames for frame in nf.correspondent_frames if frame.registered] + list(frames)))
            landmarks = list(get_unique([landmark for kf in local_bundle for landmark in kf.landmarks]))

            errors.append(self.adjust_bundle(pose_graph, local_bundle, self.config.ba_unstable_tracks))

            # trim inconsistent observations off related tracks
            for landmark in landmarks:
                self.trim_landmark(landmark, lm_min_track_len)

            landmarks = list(get_unique([landmark for kf in local_bundle for landmark in kf.landmarks]))

            # extend related tracks
            for landmark in landmarks:
                self.grow_landmark(landmark)

            landmarks = self.merge_landmarks(landmarks)

            # with Pool(8) as pool :
            #     for kf in frames:
            #         self.triangulator.triangulate_frame_features(kf, in_place=True, thread_pool=pool)

            if len(landmarks) == 0:
                break

        return errors

    def adjust_bundle(self, pose_graph: PoseGraph, frames: Sequence[KeyFrame] = None, short_tracks=False) -> Tuple[float, float]:
        frames = pose_graph.keyframes if frames is None else frames
        frames = sorted(frames, key=lambda kf: kf.idx)
        fail = np.nan, np.nan
        if len(frames) < 2:
            return fail

        # fea_noise = noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(1.345), noiseModel.Unit.Create(2))
        fea_noise = noiseModel.Isotropic.Sigma(2, 1)
        cam_cal = gtsam.Cal3_S2(self.K[0, 0], self.K[1, 1], self.K[0, 1], self.K[0, 2], self.K[1, 2])

        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        kf_indices = [kf.idx for kf in frames]
        for landmark in get_unique(chain(*(frame.landmarks for frame in frames))):
            if landmark.track_support_length < 3 and not short_tracks:
                continue

            initial_estimate.insert(gtss.L(landmark.idx), gtsam.Point3(landmark.position))

            # add each correspondence of the track to graph
            for feature in landmark.track:
                fea_frame = feature.keyframe
                reproj_f = gtsam.GenericProjectionFactorCal3_S2(
                    feature.xy, fea_noise, gtss.X(fea_frame.idx), gtss.L(landmark.idx), cam_cal)
                graph.add(reproj_f)

                # in case the track extend outside the bundle, add those frames but fix
                if fea_frame.idx not in kf_indices or pose_graph.is_origin(fea_frame):
                    fix_frame(graph, fea_frame)
                add_frame_to_estimate(initial_estimate, fea_frame)

        # if origin is involved, fix x of first non-origin frame
        # otherwise (origin not in frames), fix first and x of second
        involved_frames = [frame for frame in frames if graph.keys().count(gtss.X(frame.idx))]
        if len(involved_frames) < 2:
            return fail
        if gtss.X(pose_graph.origin_frame.idx) in initial_estimate.keys():
            fix_x_frame = next(frame for frame in involved_frames if not pose_graph.is_origin(frame))
        else:
            fix_frame(graph, involved_frames[0])
            fix_x_frame = involved_frames[1]
        fix_frame(graph, fix_x_frame, mode='x-coord')

        params = gtsam.LevenbergMarquardtParams().CeresDefaults()
        params.setRelativeErrorTol(1e-2)
        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            result = optimizer.optimize()
        except RuntimeError as err:
            raise err
            print(err)
            return fail

        err_initial = graph.error(initial_estimate)
        err_final = graph.error(result)

        # assign results
        for key in result.keys():
            key_type, index = chr(gtsam.symbolChr(key)).upper(), gtsam.symbolIndex(key)
            if key_type == 'X':
                pose_graph.keyframes[index].pose = result.atPose3(key).matrix()
            elif key_type == 'L':
                pose_graph.landmarks[index].position = result.atPoint3(key)

        return err_initial, err_final

    # landmark ops
    def grow_landmark(self, landmark: Landmark, transitivity=2):
        to_explore = collections.deque(landmark.track)
        for _ in range(transitivity):
            to_explore_next = collections.deque()
            while len(to_explore) > 0:
                feature = to_explore.pop()
                # add rogue correspondences to landmark if ok
                for corr in feature.correspondences:
                    if not corr.has_landmark and self.triangulator.check_reprojection(corr, landmark):
                        landmark.add_feature(corr)
                        to_explore_next.append(corr)
            to_explore = to_explore_next

    def trim_landmark(self, landmark: Landmark, min_track_len: Optional[int]=None):
        '''Remove inconsistent observations from landmarks.'''
        if min_track_len is None:
            min_track_len = self.config.lm_min_track_len
        for feature, inlier in zip(landmark.track, check_landmark_reprojection(
                landmark, self.K, self.lm_merge_reproj_thresh)):
            if not inlier:
                landmark.remove_feature(feature)

        if landmark.track_support_length < min_track_len or not check_landmark_parallax(landmark, self.config.lm_max_cos_parallax):
            landmark.owner.remove_landmark(landmark)

    def merge_landmarks(self, landmarks: List[Landmark]):
        change_set: List[Landmark] = []
        to_merge: List[Tuple[Landmark, Landmark]] = []
        resulting_landmarks = landmarks.copy()
        for landmark in landmarks:
            other_lm = self._suggest_landmark_merge(landmark, change_set)
            if other_lm is not None:
                to_merge.append((landmark, other_lm))
                change_set.extend([landmark, other_lm])
        for landmark, other_lm in to_merge:
            landmark.try_merge(other_lm)
            new_pos, new_track = landmark.try_merge(other_lm)
            landmark.owner.remove_landmark(landmark)
            other_lm.owner.remove_landmark(other_lm)
            resulting_landmarks.append(landmark.owner.add_landmark(new_pos, new_track))
        return [lm for lm in resulting_landmarks if lm.in_storage]

    def _suggest_landmark_merge(self, landmark: Landmark, marked: Sequence[Landmark] = []):
        if landmark in marked:
            return None

        track = landmark.track
        new_pos, new_track, other_lm = None, None, None
        for fea in track:
            for corr_fea in fea.correspondences:
                if not corr_fea.has_landmark or corr_fea.landmark.idx == landmark.idx:
                    continue
                other_lm = corr_fea.landmark
                if other_lm in marked:
                    continue

                # do merge if check passes
                new_pos, new_track = landmark.try_merge(other_lm)
                trial = Landmark(position=new_pos, _store=landmark._store)
                trial.add_features(new_track, allow_out_of_storage=True)
                if all(check_landmark_reprojection(trial, self.K, self.lm_merge_reproj_thresh)):
                    return other_lm
        return None


def check_landmark_reprojection(landmark: Landmark, K: np.ndarray, thresh=4):
    R, t, xy = [], [], []
    for fea in landmark.track:
        R.append(fea.keyframe.R)
        t.append(fea.keyframe.t)
        xy.append(fea.xy)
    error, z = reprojection_error(np.stack(R), np.stack(t), K, landmark.position[None],
                                  np.stack(xy), return_z=True)
    return ((error ** 2).sum(axis=-1) < thresh ** 2) & (z > 0)


def check_landmark_parallax(landmark: Landmark, max_cos_parallax=0.9998):
    ray_vecs = landmark.position[None] - np.stack([fea.keyframe.t for fea in landmark.track])
    ray_vecs /= np.linalg.norm(ray_vecs, axis=-1, keepdims=True)
    cos_parallax = ray_vecs @ ray_vecs.T
    # good if any pair is non-degenerate
    return (cos_parallax < max_cos_parallax).any()

def fix_frame(graph, frame: KeyFrame, mode='completely'):
    pose_key, pose = gtss.X(frame.idx), gtsam.Pose3(frame.pose)
    if mode == 'completely':
        pose_noise = np.full(6, 1e-8)
    elif mode == 'x-coord':
        pose_noise = np.array([np.inf, np.inf, np.inf, 1e-8, np.inf, np.inf])
    graph.add(gtsam.PriorFactorPose3(pose_key, pose, noiseModel.Diagonal.Sigmas(pose_noise)))

def add_frame_to_estimate(initial_estimate, frame: KeyFrame):
    pose_key, pose = gtss.X(frame.idx), gtsam.Pose3(frame.pose)
    if pose_key not in initial_estimate.keys():
        initial_estimate.insert(pose_key, pose)
