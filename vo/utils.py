from typing import List
import cv2

import numpy as np

from utils.geometry import recover_pose_essential, solve_PnP_ransac

from .pose_graph import Feature, KeyFrame, Landmark


def estimate_pose_PnP(frame: KeyFrame, K: np.ndarray, thresh_norm=0.005):
    support_features: List[Feature] = []
    support_landmarks: List[Landmark] = []
    # find out which landmarks/correspondences to depend on
    for corr in frame.correspondences:
        # in case two ref features matched to the same new feature
        if not corr[0].has_landmark and corr[1].has_landmark:
            support_features.append(corr[0])
            support_landmarks.append(corr[1].landmark)

    initial_pose = frame.pose

    if len(support_features) < 3:
        return False, initial_pose

    success, R, t, inliers = solve_PnP_ransac(
        point3d=np.stack([lm.position for lm in support_landmarks]),
        point2d=np.stack([fea.xy for fea in support_features]),
        K=K, pose_init=initial_pose, thresh_norm=thresh_norm)

    if not success or inliers.sum() < 3:
        return False, initial_pose

    return True, np.block([[R, t[None].T], [np.array([0, 0, 0, 1])]])


def estimate_pose_essential(frame_new: KeyFrame, frame_ref: KeyFrame, K: np.ndarray, thresh_norm=0.015, dist_thresh=100):
    # retreave features
    xy_r_, xy_n_ = [], []
    for ref_fea, new_fea in frame_ref.correspondences_with(frame_new):
        xy_r_.append(ref_fea.xy)
        xy_n_.append(new_fea.xy)

    if len(xy_r_) < 5:
        return False, frame_new.pose

    xy_r, xy_n = np.stack(xy_r_), np.stack(xy_n_)
    try:
        R, t, _ = recover_pose_essential(xy_n, xy_r, K, thresh_norm=thresh_norm, dist_thresh=dist_thresh)
    except cv2.error:
        # cv2.findEssentialMat may not return a 3x3 matrix, which crashes cv2.recoverPose
        return False, frame_new.pose
    pose_cand = frame_ref.pose @ np.block([[R, t[None].T], [np.array([0, 0, 0, 1])]])
    return True, pose_cand
