from typing import Optional
import numpy as np

from utils.geometry import invert_K, reprojection_error
from vo.pose_graph import PoseGraph

from .pyvista import PVVisualizer, _auto_key


class PoseGraphVisualizer(PVVisualizer):
    def __init__(self, K: Optional[np.ndarray]=None, NED=True, show_reproj=False, show_feature=False, show_image=False) -> None:
        super().__init__(NED)
        self.K = K
        self.show_reproj = show_reproj
        self.show_feature = show_feature
        self.show_image = show_image

    @_auto_key('PG')
    def set_pose_graph(self, 
                       graph: PoseGraph, 
                       K: Optional[np.ndarray]=None,
                       scale=1, short_track=False, traj=False, 
                       traj_color="white", pose="all", key=None):
        if K is None:
            K = self.K
            if K is None:
                raise ValueError("K is not provided. Expect K is known either from the dataset or from the visualizer.")
        
        lm_pos, lm_color, lm_reproj_line, fea_pos, fea_pos_color = [], [], ([], [], []), [], []
        for landmark in graph.landmarks:
            if not short_track and landmark.track_support_length < 3:
                continue
            color_fea = []
            for fea in landmark.track:
                fea_reproj_xy = reprojection_error(fea.keyframe.R, fea.keyframe.t, K, landmark.position[None], np.zeros((1, 2)))[0]
                fea_reproj_pos = (fea.keyframe.R @ invert_K(K) @ np.block([fea_reproj_xy, 1])[None].T)[:, 0] + fea.keyframe.t
                if self.show_feature:
                    fea_pos.append((fea.keyframe.R @ invert_K(K) @ np.block([fea.xy, 1])[None].T)[:, 0] + fea.keyframe.t)
                    fea_pos_color.append([[0., 0., 1.]])
                    fea_pos.append(fea_reproj_pos)
                    fea_pos_color.append([[1., 0., 0.]])
                if fea.info is not None and isinstance(fea.info, dict) and 'fea_color' in fea.info:
                    color_fea.append(fea.info['fea_color'])
                else: # collect color from image
                    xy = fea.info['xy'].round().int()
                    frame_id = fea.frame_id
                    keyframe = graph.keyframes[frame_id]
                    img = keyframe.info['img']
                    pixel = img[:, min(xy[1], img.shape[1]-1), min(xy[0], img.shape[2]-1)]
                    color_fea.append(pixel)
                if self.show_reproj:
                    lm_reproj_line[0].append(landmark.position)
                    lm_reproj_line[1].append(fea_reproj_pos)
            lm_pos.append(landmark.position)
            lm_color.append(np.mean(color_fea, axis=0) if len(color_fea) > 0 else np.full(3, np.nan))
            lm_reproj_line[2].extend([lm_color[-1]] * landmark.track_length)

        if self.show_reproj and len(lm_reproj_line[0]) > 0:
            color = np.stack(lm_reproj_line[2]) if len(lm_reproj_line[2]) > 0 else ['green'] * len(lm_reproj_line[0])
            color = np.concatenate((color, np.full((len(color), 1), 0.5)), axis=-1)
            self.set_lines(np.stack(lm_reproj_line[0]), np.stack(lm_reproj_line[1]), color=color, key=f'{key}.L0')

        if self.show_feature and len(fea_pos) > 0:
            self.set_points(np.stack(fea_pos), fea_pos_color, point_size=8, key=f'{key}.PFEA')

        if pose.lower() == "all":
            for j, kf in enumerate(graph.keyframes):
                self.set_pose(kf.R, kf.t, scale=scale, key=f'{key}.T{j}')
                # display image if available
                if self.show_image and (kf.info is not None and isinstance(kf.info, dict) and 'img' in kf.info):
                    self.set_camera(kf.info['img'], K, R=kf.R, t=kf.t, scale=scale, key=f'{key}.C{j}')
        elif pose.lower() == "last":
            kf = graph.keyframes[-1]
            self.set_pose(kf.R, kf.t, scale=scale, key=f'{key}.T_last')
            # display image if available
            if self.show_image and (kf.info is not None and isinstance(kf.info, dict) and 'img' in kf.info):
                self.set_camera(kf.info['img'], K, R=kf.R, t=kf.t, scale=scale, key=f'{key}.C_last')

        if traj:
            self.set_trajectory(np.array([kf.t for kf in graph.keyframes]), color=traj_color, key=f'{key}.traj')


        # mask = np.isfinite().all(axis=1)
        if len(lm_pos) > 0:
            self.set_points(np.stack(lm_pos), color=np.stack(lm_color), point_size=6, key=f'{key}.P3D')
