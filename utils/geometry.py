import warnings
from typing import Optional, overload

import cv2
import kornia.geometry as kg
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from skimage.measure import ransac


class Projector():

    @staticmethod
    def pix2world(points, depth_map, poses, Ks):
        """Unprojects pixels to 3D coordinates."""
        H, W = depth_map.shape[2:]
        cam = Projector._make_camera(H, W, Ks, poses)
        depths = Projector._sample_depths(depth_map, points)
        return Projector._pix2world(points, depths, cam)

    @staticmethod
    def world2pix(points, res, poses, Ks, depth_map=None, eps=1e-2):
        """Projects 3D coordinates to screen."""
        assert len(Ks.shape) == 3
        cam = Projector._make_camera(res[0], res[1], Ks, poses)
        xy, depth = Projector._world2pix(points, cam)

        if depth_map is not None:
            depth_dst = Projector._sample_depths(depth_map, xy)
            xy[(depth < 0) | (depth > depth_dst + eps) | ((xy.abs() > 1).any(dim=-1))] = np.nan

        return xy, depth

    @staticmethod
    def _make_camera(height, width, K, pose):
        """Creates a PinholeCamera with specified intrinsics and extrinsics."""
        intrinsics = torch.eye(4, 4).to(K).repeat(len(K), 1, 1)
        intrinsics[:, 0:3, 0:3] = K

        extrinsics = torch.eye(4, 4).to(pose).repeat(len(pose), 1, 1)
        extrinsics[:, 0:4, 0:4] = pose

        height, width = torch.tensor([height]).to(K), torch.tensor([width]).to(K)

        return kg.PinholeCamera(intrinsics, extrinsics, height, width)

    @staticmethod
    def _pix2world(p, depth, cam):
        """Projects p to world coordinate.

        Args:
        p:     List of points in pixels (B, N, 2).
        depth: Depth of each point(B, N).
        cam:   Camera with batch size B

        Returns:
        World coordinate of p (B, N, 3).
        """
        p = distort_denormalize(p, cam.intrinsics[..., :3, :3])
        p_h = kg.convert_points_to_homogeneous(p)
        p_cam = kg.transform_points(cam.intrinsics_inverse(), p_h) * depth.unsqueeze(-1)
        return kg.transform_points(kg.inverse_transformation(cam.extrinsics), p_cam)

    @staticmethod
    def _world2pix(p_w, cam):
        """Projects p to normalized camera coordinate.

        Args:
        p_w: List of points in world coordinate (B, N, 3).
        cam: Camera with batch size B

        Returns:
        Normalized coordinates of p in pose cam_dst (B, N, 2) and screen depth (B, N).
        """
        proj = kg.compose_transformations(cam.intrinsics, cam.extrinsics)
        p_h = kg.transform_points(proj, p_w)
        p, d = kg.convert_points_from_homogeneous(p_h), p_h[..., 2]
        return undistort_normalize(p, cam.intrinsics[..., :3, :3]), d

    @staticmethod
    def _project_points(p, depth_src, cam_src, cam_dst):
        """Projects p visible in pose T_p to pose T_q.

        Args:
        p:                List of points in pixels (B, N, 2).
        depth:            Depth of each point(B, N).
        cam_src, cam_dst: Source and destination cameras with batch size B

        Returns:
        Normalized coordinates of p in pose cam_dst (B, N, 2).
        """
        return Projector._world2pix(Projector._pix2world(p, depth_src, cam_src), cam_dst)

    @staticmethod
    def _sample_depths(depths_map, points):
        """Samples the depth of each point in points"""
        assert depths_map.shape[:2] == (len(points), 1)
        return F.grid_sample(depths_map, points[:, None], align_corners=False)[:, 0, 0, ...]


def src_repeat(x, n_dst=None):
    """[b0 b1 b2 ...] -> [b0 b0 ... b1 b1 ...]"""
    B, shape = x.shape[0], x.shape[1:]
    n_dst = n_dst if n_dst is not None else B
    return x.unsqueeze(1).expand(B, n_dst, *shape).reshape(B * n_dst, *shape)


def dst_repeat(x, n_src=None):
    """[b0 b1 b2 ...] -> [b0 b1 ... b0 b1 ...]"""
    B, shape = x.shape[0], x.shape[1:]
    n_src = n_src if n_src is not None else B
    return x.unsqueeze(0).expand(n_src, B, *shape).reshape(n_src * B, *shape)


def coord_list_grid_sample(values, points, mode='bilinear'):
    dim = len(points.shape)
    points = points.view(values.size(0), 1, -1, 2) if dim == 3 else points
    output = F.grid_sample(values, points, mode, align_corners=True).permute(0, 2, 3, 1)
    return output.squeeze(1) if dim == 3 else output


def pose_vec2pose_mat(pose):
    """Converts pose vectors to pose matrices.
    Args:
      pose: [tx, ty, tz, qx, qy, qz, qw] (..., 7).
    Returns:
      [R t] (..., 4, 4).
    """
    t = pose[..., 0:3, None]
    R = Rotation.from_quat(pose[..., 3:7]).as_matrix().astype(np.float32)
    out = np.tile(np.eye(4, dtype=np.float32), tuple(pose.shape[:-2]) + (1, 1))
    out[..., 0:3, 0:3] = R
    out[..., 0:3, 3:4] = t
    return torch.from_numpy(out)


@overload
def invert_Rt(K: np.ndarray) -> np.ndarray:
    ...


@overload
def invert_Rt(K: torch.Tensor) -> torch.Tensor:
    ...


def invert_Rt(Rt_mat):
    R, t = Rt_mat[..., 0:3, 0:3], Rt_mat[..., 0:3, 3:4]
    reps = tuple(Rt_mat.shape[:-2]) + (1, 1)
    if isinstance(Rt_mat, np.ndarray):
        out = np.tile(np.eye(4), reps)
        RT = np.swapaxes(R, -1, -2)
    elif isinstance(Rt_mat, torch.Tensor):
        out = torch.eye(4, device=Rt_mat.device).repeat(reps)
        RT = torch.swapaxes(R, -1, -2)
    else:
        raise ValueError(f"Unknown matrix type: {type(Rt_mat)}")

    out[..., 0:3, 0:3] = RT
    out[..., 0:3, 3:4] = -RT @ t
    return out


@overload
def invert_K(K: np.ndarray) -> np.ndarray:
    ...


@overload
def invert_K(K: torch.Tensor) -> torch.Tensor:
    ...


def invert_K(K):
    if isinstance(K, np.ndarray):
        out = np.zeros_like(K)
    elif isinstance(K, torch.Tensor):
        out = torch.zeros_like(K)
    else:
        raise ValueError(f"Unknown matrix type: {type(K)}")

    fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]
    out[..., 0, 0], out[..., 1, 1] = 1. / fx, 1. / fy
    out[..., 0, 2], out[..., 1, 2] = - cx * out[..., 0, 0], - cy * out[..., 1, 1]
    out[..., 2, 2] = 1.
    return out


def reprojection_error(R, t, K, points, fea_xy, eps=1e-6, return_z=False):
    if R.ndim == 2:
        R = R[None]
    if K.ndim == 2:
        K = K[None]
    if isinstance(t, np.ndarray):
        t = np.atleast_2d(t)
    elif isinstance(t, torch.Tensor):
        t = torch.atleast_2d(t)

    xy_mh = ((K @ R.swapaxes(-1, -2)) @ (points - t)[:, :, None])[:, :, 0]
    xy_m = xy_mh[:, :2] / (xy_mh[:, 2:3] + eps)
    if return_z:
        return xy_m - fea_xy, xy_mh[:, 2]
    else:
        return xy_m - fea_xy


def denorm_thresh(thresh_norm: float, K: np.ndarray) -> float:
    return max(K[0, 0], K[1, 1]) * thresh_norm


def recover_pose_essential(xy_n: np.ndarray, xy_r: np.ndarray, K: np.ndarray, thresh_norm=0.0003, dist_thresh=100):
    thresh = denorm_thresh(thresh_norm, K)

    e_mat, mask = cv2.findEssentialMat(xy_n, xy_r, K, threshold=thresh)
    _, R, t, _, _ = cv2.recoverPose(e_mat, xy_n, xy_r, K, distanceThresh=dist_thresh, mask=mask)
    mask = mask.flatten().astype(bool)
    t = t.flatten()

    # get inliers by evaluating reprojection error
    pose_ref = np.eye(4)
    pose_cand = np.block([[R, t[None].T], [np.array([0, 0, 0, 1])]])
    positions, _ = triangulate(pose_cand, xy_n, pose_ref, xy_r, K)
    err_r, z_r = reprojection_error(pose_ref[0:3, 0:3], pose_ref[0:3, 3], K, positions, xy_r, return_z=True)
    err_n, z_n = reprojection_error(R, t, K, positions, xy_n, return_z=True)
    err = (np.linalg.norm(err_r, axis=-1) + np.linalg.norm(err_n, axis=-1)) / 2
    inliers = (err < thresh) & (z_r > 0) & (z_n > 0)
    return R, t, inliers


def solve_PnP_ransac(point3d: np.ndarray, point2d: np.ndarray, K: np.ndarray, pose_init: np.ndarray = None, thresh_norm=1.0, refine=True):
    success = False
    if len(point3d) >= 5:
        if pose_init is None:
            rvec, tvec = None, None
        else:
            RT = Rotation.from_matrix(pose_init[0:3, 0:3].T)
            rvec, tvec = RT.as_rotvec(), -RT.apply(pose_init[0:3, 3])

        thresh = denorm_thresh(thresh_norm, K)
        success, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
            point3d, point2d, K, distCoeffs=None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            reprojectionError=thresh)

    if not success or inlier_idx is None:
        return False, None, None, None

    inliers = np.zeros(len(point2d), dtype=bool)
    inliers[inlier_idx.flatten()] = True

    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(point3d[inliers], point2d[inliers], K, None, rvec, tvec)
    R, RTt = Rotation.from_rotvec(-rvec.flatten()), -tvec.flatten()
    return True, R.as_matrix(), R.apply(RTt), inliers


def triangulate(pose0: np.ndarray, xy0: np.ndarray, pose1: np.ndarray, xy1: np.ndarray, K: np.ndarray):
    proj_0, proj_1 = K @ invert_Rt(pose0)[:3], K @ invert_Rt(pose1)[:3]
    pts_h = cv2.triangulatePoints(proj_0, proj_1, xy0.T, xy1.T).T
    pts_h /= pts_h[:, 3:4]
    pts: np.ndarray = pts_h[:, :3]
    # remove points with z < 0
    inlier: np.ndarray = ((proj_0 @ pts_h.T)[2] > 0) & ((proj_1 @ pts_h.T)[2] > 0)
    return pts, inlier


def triangulate_multiview(poses: np.ndarray, xys: np.ndarray, K: np.ndarray):
    R, t = poses[:, 0:3, 0:3], poses[:, 0:3, 3:4]
    RT = np.swapaxes(R, -1, -2)
    proj = K[None] @ np.concatenate((RT, -RT @ t), axis=2)  # nx3x4

    # https://github.com/strawlab/pymvg/blob/2b99ccb459063f34dbe801bdbbfcf1209b1fb3e5/pymvg/multi_camera_system.py#L198-L226
    xP = xys[:, :, None] * proj[:, 2:3, :]  # nx2x1 * nx1x4
    A = (xP - proj[:, 0:2, :]).reshape(-1, 4)  # (n*2)x4
    _, v = np.linalg.eigh(A.T @ A)
    pts_h = v[:, 0] / v[3:4, 0]  # 4x1

    return pts_h[:3], (proj @ pts_h)[:, 2] > 0


class TriangulationEstimator:
    def __init__(self, K) -> None:
        self.K = K
        self.position = None

    def estimate(self, poses: np.ndarray, xys: np.ndarray):
        self.position, inlier = triangulate_multiview(poses, xys, self.K)
        return inlier.all()

    def residuals(self, poses: np.ndarray, xys: np.ndarray):
        err = reprojection_error(poses[:, 0:3, 0:3], poses[:, 0:3, 3], self.K, self.position[None], xys)
        return np.linalg.norm(err, axis=-1)


def triangulate_ransac(poses: np.ndarray, xys: np.ndarray, K: np.ndarray, thresh=1, max_trials=100):
    def model_class(): return TriangulationEstimator(K)
    with warnings.catch_warnings():
        # mute "UserWarning: No inliers found."
        warnings.simplefilter("ignore")
        model, inlisers = ransac((poses, xys), model_class, min_samples=2,
                                 residual_threshold=thresh, max_trials=max_trials)
    if inlisers is not None:
        return model.position, inlisers
    else:
        return np.full(3, np.nan), np.zeros(len(poses), dtype=bool)


@overload
def fundamental_matrix(T0: np.ndarray, K0: np.ndarray, T1: np.ndarray, K1: np.ndarray) -> np.ndarray:
    ...


@overload
def fundamental_matrix(T0: torch.Tensor, K0: torch.Tensor, T1: torch.Tensor, K1: torch.Tensor) -> torch.Tensor:
    ...


def fundamental_matrix(T0, K0, T1, K1):
    R0, t0, R1, t1 = T0[..., 0:3, 0:3], T0[..., 0:3, 3], T1[..., 0:3, 0:3], T1[..., 0:3, 3]

    if isinstance(t0, np.ndarray):
        backend = np
    elif isinstance(t0, torch.Tensor):
        backend = torch
    else:
        raise ValueError(f"Unknown matrix type: {type(t0)}")

    e = (K0 @ R0.swapaxes(-1, -2) @ backend.atleast_2d(t1 - t0)[..., None])[..., 0]
    e_cross = backend.zeros_like(R0)
    e_cross[..., 0, 1], e_cross[..., 0, 2], e_cross[..., 1, 2] = -e[..., 2], e[..., 1], -e[..., 0]
    e_cross[..., 1, 0], e_cross[..., 2, 0], e_cross[..., 2, 1] = e[..., 2], -e[..., 1], e[..., 0]

    # return invert_K(K1).swapaxes(-1, -2) @ (R1.swapaxes(-1, -2) @ R0) @ K0.swapaxes(-1, -2) @ e_cross
    skew = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    pose2fund = lambda K1, K2, R, t: np.linalg.inv(K2).T @ R @ K1.T @ skew((K1 @ R.T).dot(t.reshape(3,)))
    return pose2fund(K0, K1, (R1.swapaxes(-1, -2) @ R0), t1-t0)

def distort_denormalize(
        coords: torch.Tensor, K: torch.Tensor, dist_coeff: Optional[torch.Tensor] = None, denormalize: bool = True):
    if dist_coeff is None:
        dist_coeff = torch.zeros(4)

    new_K = torch.eye(3).to(coords) if denormalize else None

    return kg.distort_points(coords, K.to(coords), dist_coeff.to(coords), new_K=new_K)


def undistort_normalize(
        coords: torch.Tensor, K: torch.Tensor, dist_coeff: Optional[torch.Tensor] = None, normalize: bool = True):
    if dist_coeff is None:
        dist_coeff = torch.zeros(4)
        num_iters = 0
    else:
        num_iters = 5

    new_K = torch.eye(3).to(coords) if normalize else None

    return kg.undistort_points(coords, K.to(coords), dist_coeff.to(coords), new_K=new_K, num_iters=num_iters)
