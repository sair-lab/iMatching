from functools import wraps
import threading
import time

import matplotlib.colors as mc

import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
import torch


def _named_color(c):
    if isinstance(c, str) and c in mc.get_named_colors_mapping():
        return mc.to_rgba_array(c)[0, :3]
    elif isinstance(c, np.ndarray) and c.shape == (3,):
        return c
    else:
        raise ValueError(f"Unrecognized color {c}")

def make_pose_geom(c_color="w", x_color="r", y_color="g", z_color="b"):
    origin = pv.Sphere(0.1)
    origin["color"] = np.tile(_named_color(c_color), (origin.n_points, 1))
    x_axis = pv.Arrow(direction=(1, 0, 0))
    x_axis["color"] = np.tile(_named_color(x_color), (x_axis.n_points, 1))
    y_axis = pv.Arrow(direction=(0, 1, 0))
    y_axis["color"] = np.tile(_named_color(y_color), (y_axis.n_points, 1))
    z_axis = pv.Arrow(direction=(0, 0, 1))
    z_axis["color"] = np.tile(_named_color(z_color), (z_axis.n_points, 1))
    axes = origin.merge(x_axis).merge(y_axis).merge(z_axis)
    return axes


def _make_point_label(label, location=[0., 0., 0.]):
    point = pv.PolyData([[0., 0., 0.]])
    point['label'] = [label]
    return point


def _auto_key(key_label: str):
    def _register(method):
        @wraps(method)
        def _wrapped_method(*args, key=None, **kwargs):
            '''Increments key once called'''
            key_count = PVVisualizer._key_counter.setdefault(method, 0)
            do_inc = False
            if key is None:
                do_inc = True
                key = f'{key_label}{key_count}'
            ret = method(*args, key=str(key), **kwargs)
            if do_inc:
                PVVisualizer._key_counter[method] += 1
            return ret
        return _wrapped_method
    return _register


class PVVisualizer():
    POSE_GEOM = make_pose_geom()
    _key_counter = {}

    def __init__(self, NED=True) -> None:
        self.plotter = pv.Plotter()
        self.plotter.show_axes()
        self.plotter.add_axes_at_origin(labels_off=True)
        self.geometries = {}
        self.lock = threading.Lock()

    @_auto_key('P')
    def set_points(self, pts, color=None, point_size=5., key=None):
        if len(pts) < 1:
            return
        point_cloud = pv.PolyData(pts)
        point_cloud["color"] = np.ones((len(pts), 3)) if color is None else color

        self._add_geometry(key, point_cloud, lambda: self.plotter.add_mesh(
            point_cloud, scalars="color", rgb=True, point_size=point_size,
            render_points_as_spheres=True))

    @_auto_key('C')
    def set_camera(self, img, K, R=None, t=None, quat=None, pose7=None, scale=1, key=None):
        pose_mat = self.pose_mat(R=R, t=t, quat=quat, pose7=pose7)
        z_hat = pose_mat[0:3, 2] / np.linalg.norm(pose_mat[0:3, 2])
        image_plane = pv.Plane(i_size=img.shape[-1] / K[0, 0], j_size=img.shape[-2] / K[1, 1])
        image_plane = image_plane.scale([scale] * 3, inplace=False) \
            .transform(pose_mat, inplace=True).translate(z_hat * scale, inplace=False)

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # append alpha channel
        img = np.concatenate((img, np.full((1, img.shape[-2], img.shape[-1]), 0.75)))
        # convert to HWC, flip uv as we are looking at from behind
        img = (img.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
        tex = pv.numpy_to_texture(np.flip(img, (0, 1)))

        def modify_fn(geom_actor):
            # set both geometry and texture
            geom_actor[0].copy_from(image_plane)
            geom_actor[1].SetTexture(tex)
            return geom_actor

        self._add_geometry(key, image_plane, lambda: self.plotter.add_mesh(image_plane, texture=tex), modify_fn)

    @_auto_key('T')
    def set_pose(self, R=None, t=None, quat=None, pose7=None, scale=1, colors=None, key=None):
        pose_mat = self.pose_mat(R=R, t=t, quat=quat, pose7=pose7)

        geom = self.POSE_GEOM if colors is None else make_pose_geom(*colors)
        pose = geom.scale([scale] * 3, inplace=False).transform(pose_mat)
        # label = _make_point_label(key).scale([scale] * 3, inplace=False).transform(pose_mat)
        self._add_geometry(key, pose, lambda: self.plotter.add_mesh(pose, scalars="color", rgb=True))
        # self._add_geometry(f'{key}_label', label,
        #                    lambda: self.plotter.add_point_labels(label, 'label', point_size=1, font_size=36))

    @_auto_key('R')
    def set_trajectory(self, ts=None, tube=True, radius=0.01, color="white", key=None):
        traj = pv.Spline(ts)
        if tube:
            traj = traj.tube(radius=radius)
        traj["color"] = np.tile(_named_color(color), (traj.n_points, 1))

        def create_fn():
            if traj.n_points == 0:
                # create again on next modify_fn call
                return None
            return self.plotter.add_mesh(traj, scalars="color", rgb=True)

        self._add_geometry(key, traj, create_fn)

    def run(self, callback=None, fr=20):
        self.plotter.show(interactive_update=True)

        def anim_loop(diff_t):
            if callback is not None:
                callback(diff_t)
            with self.lock:
                self.plotter.update()

        def loop_main():
            last = time.time()
            while True:
                cur = time.time()
                anim_loop(cur - last)
                last = cur
                time.sleep(1 / fr)

        threading.Thread(target=loop_main).start()

        # def _set_interval(sec, last_t):
        #     def loop_wrapper():
        #         cur_t = time.time()
        #         _set_interval(sec, cur_t)
        #         anim_loop(cur_t - last_t)
        #     t = threading.Timer(sec, loop_wrapper)
        #     t.start()
        #     return t

        # _set_interval(1 / fr, time.time())

    @_auto_key('L')
    def set_lines(self, x0, x1, color=None, key=None):
        # [(x0, y0), (x0', y0'), (x1, y1), (x1', y1'), ...]
        points = np.empty((len(x0) + len(x1), 3), dtype=x0.dtype)
        points[0::2] = x0
        points[1::2] = x1

        line_bundle = pv.line_segments_from_points(points)
        line_bundle["color"] = color
        self._add_geometry(key, line_bundle, lambda: self.plotter.add_mesh(line_bundle, rgb=True))

    def _add_geometry(self, key, geom, create_fn, modify_fn=None):
        with self.lock:
            if key not in self.geometries:
                self.geometries[key] = (geom, create_fn())
            else:
                # add_mesh is skipped for some reason (e.g. n_points == 0)
                if self.geometries[key][1] is None:
                    self.geometries[key] = (geom, create_fn())
                elif modify_fn is not None:
                    self.geometries[key] = modify_fn(self.geometries[key])
                else:
                    self.geometries[key][0].copy_from(geom)

    def pose_mat(self, R=None, t=None, quat=None, pose7=None):
        if pose7 is not None:
            return self.pose_mat(quat=pose7[..., 0:4], t=pose7[..., 4:7])
        elif quat is not None:
            # o3d follows Eigen's wxyz convention
            R = Rotation.from_quat(np.array(quat))
        t = t[:, None]
        return np.block([[R, t], [np.array([0, 0, 0, 1])]])
