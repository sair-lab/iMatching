from __future__ import annotations
from collections import deque

from dataclasses import InitVar, field
from itertools import chain
from typing import Any, List, Sequence, Tuple
from typing_extensions import Self
import numpy as np

from vo.pose_graph.indexd_store import get_unique

from . import FeaId, Corrs, FeaIdList, CorrsTable, FeaIdTable, INVALID_INDEX, _IndexMixin, IndexedStore, indexed_dataclass


@indexed_dataclass(field_exclude=['_frame_id'])
class Feature(_IndexMixin['Feature', 'PoseGraph']):

    xy: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    _correspondences: FeaIdList = field(default_factory=list)
    _frame_id: int = INVALID_INDEX
    _landmark_id: int = INVALID_INDEX
    info: Any = None

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def feature_id(self) -> FeaId:
        return self._frame_id, self.idx

    @property
    def correspondences(self):
        return (self.owner.get_feature(frame_id, feature_id) for frame_id, feature_id in self._correspondences)

    @property
    def n_correspondences(self):
        return len(self._correspondences)

    @property
    def connected_features(self) -> List[Feature]:
        return self.get_connected_features(-1)

    @property
    def keyframe(self) -> KeyFrame:
        return self.owner.keyframes[self._frame_id]

    @property
    def landmark(self) -> Landmark:
        return self.owner.landmarks[self._landmark_id]

    @property
    def has_landmark(self) -> bool:
        return self._landmark_id != INVALID_INDEX

    def get_connected_features(self, transitivity=-1) -> List[Feature]:
        # bfs
        ids_to_explore = deque([fea.feature_id for fea in self.correspondences])
        ids_explored = [self.feature_id]
        features = [self]
        ttl = transitivity
        while ttl != 0 and len(ids_to_explore) > 0:
            ids_to_explore_next: deque[FeaId] = deque()
            while len(ids_to_explore) > 0:
                frame_id, feature_id = ids_to_explore.popleft()
                if (frame_id, feature_id) not in ids_explored:
                    feature = self.owner.get_feature(frame_id, feature_id)
                    features.append(feature)
                    ids_to_explore_next.extend([fea.feature_id for fea in feature.correspondences])
                    ids_explored.append((frame_id, feature_id))
            ids_to_explore = ids_to_explore_next
            ttl -= 1
        return features

    def add_correspondence(self, other: Feature):
        if other.feature_id not in self._correspondences:
            self._correspondences.append(other.feature_id)

    def del_correspondence(self, other: Feature):
        if other.feature_id in self._correspondences:
            self._correspondences.remove(other.feature_id)

    def set_landmark(self, landmark: Landmark):
        self._landmark_id = landmark.idx if landmark is not None else INVALID_INDEX


@indexed_dataclass()
class KeyFrame(_IndexMixin['KeyFrame', 'PoseGraph']):

    pose: np.ndarray = field(default_factory=lambda: np.full((4, 4), np.nan))
    _K: InitVar[np.ndarray] = None
    _cam_mat: np.ndarray = field(default_factory=lambda: np.array([1., 1., 0., 0.]))
    _features: IndexedStore[Feature, PoseGraph] = None
    _registered: bool = False
    _corrs_by_frame: CorrsTable = field(default_factory=dict)
    info: Any = None

    def __post_init__(self, _K: np.ndarray = None):
        if _K is not None:
            self._cam_mat[0], self._cam_mat[1] = _K[0, 0], _K[1, 1]
            self._cam_mat[2], self._cam_mat[3] = _K[0, 2], _K[1, 2]

    @property
    def R(self) -> np.ndarray:
        return self.pose[0:3, 0:3]

    @property
    def t(self) -> np.ndarray:
        return self.pose[0:3, 3]

    @property
    def K(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1] = self._cam_mat[0], self._cam_mat[1]
        K[0, 2], K[1, 2] = self._cam_mat[2], self._cam_mat[3]
        return K

    @property
    def features(self) -> IndexedStore[Feature, PoseGraph]:
        return self._features

    @features.setter
    def features(self, new_feature):
        self._features = new_feature

    @property
    def correspondent_frames(self):
        return (self.owner.keyframes[frame_id]
                for frame_id in self._corrs_by_frame.keys())

    @property
    def n_corrs_by_frame(self):
        return [(self.owner.keyframes[frame_id], len(corrs)) for frame_id, corrs in self._corrs_by_frame.items()]

    @property
    def connected_frames(self):
        return get_unique(self.owner.keyframes[frame_id]
                          for landmark in self.landmarks
                          for frame_id in landmark._track.keys())

    @property
    def correspondences(self) -> List[Tuple[Feature, Feature]]:
        return sum([self.correspondences_with(frame) for frame in self.correspondent_frames], [])

    @property
    def correspondent_features(self):
        return get_unique(self._features[corr[0]]
                          for corrs in self._corrs_by_frame.values()
                          for corr in corrs)

    @property
    def correspondent_features_proper(self):
        return get_unique(self._features[corr[0]]
                          for i, corrs in self._corrs_by_frame.items()
                          if i != self.idx for corr in corrs)

    @property
    def landmarks(self) -> List[Landmark]:
        return [fea.landmark for fea in self._features if fea.has_landmark]

    @property
    def registered(self) -> bool:
        return self._registered

    def correspondences_with(self, other: KeyFrame) -> List[Tuple[Feature, Feature]]:
        if other.idx not in self._corrs_by_frame:
            return []
        return [(self._features[self_id], other._features[other_id])
                for self_id, other_id in self._corrs_by_frame[other.idx]]

    def add_features(self, xy: np.ndarray, info: Any = None):
        self._features.add(xy=xy, info=info)

    def add_correspondences(self, other: KeyFrame, correspondences: Sequence[Corrs]):
        for fea_self_id, fea_other_id in correspondences:
            fea_self, fea_other = self._features[fea_self_id], other._features[fea_other_id]
            fea_self.add_correspondence(fea_other)
            fea_other.add_correspondence(fea_self)
            self._add_correspondent_feature_id(fea_self.feature_id, fea_other.feature_id)
            other._add_correspondent_feature_id(fea_other.feature_id, fea_self.feature_id)

    def del_correspondences(self, other: KeyFrame):
        for fea_self, fea_other in list(self.correspondences_with(other)):
            self._del_correspondent_feature_id(fea_self.feature_id, fea_other.feature_id)
            # "double-free" prevention
            if self.idx != other.idx:
                other._del_correspondent_feature_id(fea_other.feature_id, fea_self.feature_id)
            fea_self.del_correspondence(fea_other)
            if self.idx != other.idx:
                fea_other.del_correspondence(fea_self)

    def _add_correspondent_feature_id(self, self_feature_id: FeaId, other_feature_id: FeaId):
        assert self_feature_id[0] == self.idx
        self._corrs_by_frame.setdefault(other_feature_id[0], []).append((self_feature_id[1], other_feature_id[1]))

    def _del_correspondent_feature_id(self, self_feature_id: FeaId, other_feature_id: FeaId):
        assert self_feature_id[0] == self.idx
        self._corrs_by_frame[other_feature_id[0]].remove((self_feature_id[1], other_feature_id[1]))
        if len(self._corrs_by_frame[other_feature_id[0]]) == 0:
            self._corrs_by_frame.pop(other_feature_id[0])

    def register(self):
        self._registered = True

    def unregister(self):
        self._registered = False


@indexed_dataclass()
class Landmark(_IndexMixin['Landmark', 'PoseGraph']):

    position: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    _track: FeaIdTable = field(default_factory=dict)

    @property
    def track(self) -> List[Feature]:
        return [self.owner.get_feature(frame_id, feature_id) for frame_id, feature_id in chain(*self._track.values())]

    @property
    def track_length(self) -> int:
        return sum(len(row) for row in self._track.values())

    @property
    def track_support_length(self) -> int:
        return len(self._track.keys())

    def features_on(self, frame: KeyFrame):
        if frame.idx not in self._track:
            return []
        return [self.owner.get_feature(frame_id, feature_id) for frame_id, feature_id in self._track[frame.idx]]

    def add_feature(self, feature: Feature, allow_out_of_storage=False):
        if not allow_out_of_storage:
            self._assert_in_storage()

        feature._assert_in_storage()
        feature_id = feature.feature_id

        track_of_frame = self._track.setdefault(feature_id[0], [])
        if feature_id not in track_of_frame:
            track_of_frame.append(feature_id)
            if self.in_storage:
                feature.set_landmark(self)

    def add_features(self, features: Sequence[Feature], allow_out_of_storage=False):
        for feature in features:
            self.add_feature(feature, allow_out_of_storage)

    def remove_feature(self, feature: Feature, allow_out_of_storage=False):
        if not allow_out_of_storage:
            self._assert_in_storage()

        feature._assert_in_storage()
        feature_id = feature.feature_id

        # remove feature from track and clear row if needed
        if self.in_storage:
            feature.set_landmark(None)
        track_on_frame = self._track[feature_id[0]]
        track_on_frame.remove(feature_id)
        if len(track_on_frame) == 0:
            self._track.pop(feature_id[0])

    def try_merge(self, other: Landmark):
        # mean weighted by track length
        len_s, len_o = self.track_length, other.track_length
        merged_pos = (self.position * len_s + other.position * len_o) / (len_s + len_o)
        merged_track = self.track + other.track
        return merged_pos, merged_track

    def track_contains(self, feature: Feature):
        return feature.frame_id in self._track and feature.feature_id in self._track[feature.frame_id]


class PoseGraph():
    def __init__(self) -> None:
        self._keyframes = IndexedStore(KeyFrame, self)
        self._landmarks = IndexedStore(Landmark, self)
        self._origin_idx = INVALID_INDEX

    @property
    def features(self) -> Sequence[IndexedStore[Feature, Self]]:
        return [kf.features for kf in self._keyframes]

    @property
    def keyframes(self):
        return self._keyframes

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def origin_frame(self):
        return self._keyframes[self._origin_idx] if self._origin_idx != INVALID_INDEX else None

    def add_keyframe(self, pose: np.ndarray = None, fea_xy: Sequence[np.ndarray] = None,
                     K: np.ndarray = None, fea_info: Sequence = None, frame_info: Any = None) -> KeyFrame:
        self._keyframes.add(single=True, pose=pose, features=None, _K=K, info=frame_info)
        new_frame = self._keyframes.iloc[-1]
        new_features = IndexedStore(Feature, self, {'_frame_id': new_frame.idx})
        if fea_xy is not None:
            new_features.add(xy=fea_xy, info=fea_info)
        new_frame.features = new_features
        return new_frame

    def add_correspondences(self, keyframe0: KeyFrame, keyframe1: KeyFrame, matches: Sequence[Corrs]):
        keyframe0.add_correspondences(keyframe1, matches)

    def add_landmark(self, initial_pos: np.ndarray = None, track: Sequence[Feature] = []) -> Landmark:
        if initial_pos is None:
            initial_pos = np.full(3, np.nan)
        self._landmarks.add(True, position=initial_pos)
        new_landmark = self._landmarks.iloc[-1]
        new_landmark.add_features(track)
        return new_landmark

    def get_feature(self, frame_id: int, feature_id: int) -> Feature:
        return self._keyframes[frame_id].features[feature_id]

    def remove_landmark(self, landmark: Landmark):
        landmark._assert_in_storage()
        for feature in landmark.track:
            feature.set_landmark(None)
        self._landmarks.remove(landmark)

    def remove_keyframe(self, keyframe: KeyFrame):
        keyframe._assert_in_storage()

        # remove feature from landmarks
        for lm in list(keyframe.landmarks):
            for fea in lm.features_on(keyframe):
                lm.remove_feature(fea)
            if lm.track_support_length < 2 and lm.in_storage:
                lm.owner.remove_landmark(lm)

        # chop-off corrs with other frames
        for corr_kf in list(keyframe.correspondent_frames):
            keyframe.del_correspondences(corr_kf)

        self._keyframes.remove(keyframe)

    def correspondences_between(self, keyframe0: KeyFrame, keyframe1: KeyFrame):
        return keyframe0.correspondences_with(keyframe1)

    def set_origin(self, frame: KeyFrame):
        self._origin_idx = frame.idx if frame is not None else INVALID_INDEX

    def is_origin(self, frame: KeyFrame):
        return self._origin_idx != INVALID_INDEX and self._origin_idx == frame.idx
