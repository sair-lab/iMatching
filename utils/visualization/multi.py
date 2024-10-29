#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from matplotlib import cm
import matplotlib.colors as mc
from matplotlib import pyplot as plt
import kornia.geometry.conversions as C
from matplotlib.animation import FuncAnimation


class MultiVisualizer():
    vis_id = 0

    def __init__(self, display='imshow', default_name=None, **kwargs):
        self.radius, self.thickness = 1, 1
        self.default_name = 'Visualizer %d' % self.vis_id if default_name is None else default_name
        MultiVisualizer.vis_id += 1

        if display == 'imshow':
            self.displayer = ImshowDisplayer()
        if display == 'plt':
            self.displayer = PltDisplayer(**kwargs)
        elif display == 'tensorboard':
            self.displayer = TBDisplayer(**kwargs)
        elif display == 'video':
            self.displayer = VideoFileDisplayer(**kwargs)
        elif display == 'gif':
            self.displayer = GIFFileDisplayer(**kwargs)

    def show(self, images, points=None, color='red', nrow=4, values=None, vmin=None, vmax=None, name=None, step=0):
        b, c, h, w = images.shape
        if c == 3:
            images = torch2cv(images)
        elif c == 1:  # show colored values
            images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
            images = get_colors(color, images.squeeze(-1), vmin, vmax)

        if points is not None:
            points = C.denormalize_pixel_coordinates(points, h, w).to(torch.int)
            for i, pts in enumerate(points):
                colors = get_colors(color, [0]*len(pts) if values is None else values[i], vmin, vmax)
                images[i] = circles(images[i], pts, self.radius, colors, self.thickness)

        disp_name = name if name is not None else self.default_name

        if nrow is not None:
            images = torch.tensor(images.copy()).permute((0, 3, 1, 2))
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1).permute((1, 2, 0))
            self.displayer.display(disp_name, grid.numpy(), step)
        else:
            for i, img in enumerate(images):
                self.displayer.display(disp_name, img, step)

    # def showmatch(self, imges1, points1, images2, points2, color='blue', values=None, vmin=None, vmax=None, name=None, step=0, nrow=2):
    #     match_pairs = []
    #     for i, (img1, pts1, img2, pts2) in enumerate(zip(imges1, points1, images2, points2)):
    #         assert len(pts1) == len(pts2)
    #         h, w = img1.size(-2), img1.size(-1)
    #         pts1 = C.denormalize_pixel_coordinates(pts1, h, w)
    #         pts2 = C.denormalize_pixel_coordinates(pts2, h, w)
    #         img1, img2 = torch2cv(torch.stack([img1, img2]))
    #         colors = get_colors(color, [0]*len(pts1) if values is None else values[i], vmin, vmax)
    #         match_pairs.append(torch.tensor(matches(img1, pts1, img2, pts2, colors)))

    #     images = torch.stack(match_pairs).permute((0, 3, 1, 2))
    #     grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1).permute((1, 2, 0))
    #     self.displayer.display(name if name is not None else self.default_name, grid.numpy(), step)

    def showmatch(
            self, imges1, points1, images2, points2, m_indices=None, m_sizes=None, color='blue', values=None, vlim=[None, None],
            pts_colors=None, pts_values=None, pts_vlim=None, name=None, step=0, nrow=2, alpha=1):
        pts_colors = [color] * 2 if pts_colors is None else pts_colors
        pts_values = [values] * 2 if pts_values is None else pts_values
        pts_vlim = [vlim] * 2 if pts_vlim is None else pts_vlim

        if m_sizes is None:
            m_sizes = torch.zeros(len(imges1), dtype=int)
        else:
            m_sizes = torch.tensor(m_sizes)

        if m_indices is None:
            m_indices = torch.cat([torch.arange(m_size, dtype=int).expand(2, m_size).T for m_size in m_sizes])

        line_colors = self._get_colors(color, values, vlim, m_indices.shape[:-1])
        pts1_colors = [self._get_colors(pts_colors[0], pts_values[0], pts_vlim[0], points1_.shape[:-1]) for points1_ in points1]
        pts2_colors = [self._get_colors(pts_colors[1], pts_values[1], pts_vlim[1], points2_.shape[:-1]) for points2_ in points2]

        match_pairs = []
        b_end = torch.cumsum(m_sizes, dim=0)
        b_start = torch.cat((torch.zeros(1).to(b_end), b_end[:-1]))
        for i, (img1, pts1, img2, pts2, b_st, b_nd) in enumerate(zip(imges1, points1, images2, points2, b_start, b_end)):
            # assert len(pts1) == len(pts2)
            h, w = img1.size(-2), img1.size(-1)
            # pts1 = C.denormalize_pixel_coordinates(pts1, h, w)
            # pts2 = C.denormalize_pixel_coordinates(pts2, h, w)
            img1, img2 = torch2cv(torch.stack([img1, img2]))
            match_pairs.append(torch.tensor(matches(img1, pts1, img2, pts2, m_indices[b_st:b_nd],
                                                    line_colors[b_st:b_nd], (pts1_colors[i], pts2_colors[i]), alpha=alpha)))

        images = torch.stack(match_pairs).permute((0, 3, 1, 2))
        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1).permute((1, 2, 0))
        self.displayer.display(name if name is not None else self.default_name, grid.numpy(), step)

    @staticmethod
    def _get_colors(name, values, lim, default_shape):
        if values is None:
            default_value = (lim[0] + lim[1]) / 2 if lim[0] is not None and lim[1] is not None else np.nan
            values = np.ones(default_shape) * default_value
        return get_colors(name, values, lim[0], lim[1])

    def close(self):
        self.displayer.close()


class VisDisplayer():
    def display(self, name, frame, step=0):
        raise NotImplementedError()

    def close(self):
        pass


class PltDisplayer(VisDisplayer):
    def __init__(self, fig_size=None):
        self.fig_size = fig_size

    def display(self, name, frame, step=0):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        ax.imshow(frame[:, :, ::-1])
        fig.tight_layout()

    def close(self):
        plt.close('all')


class ImshowDisplayer(VisDisplayer):
    def display(self, name, frame, step=0):
        cv2.imshow(name, frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class TBDisplayer(VisDisplayer):
    def __init__(self, writer):
        self.writer = writer

    def display(self, name, frame, step=0):
        self.writer.add_image(name, frame[:, :, ::-1], step, dataformats='HWC')


class VideoFileDisplayer(VisDisplayer):
    def __init__(self, save_dir=None, framerate=10):
        if save_dir is None:
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.save_dir = os.path.join('.', 'vidout', current_time)
        else:
            self.save_dir = save_dir
        self.framerate = framerate
        self.writer = {}

    def display(self, name, frame, step=0):
        if name not in self.writer:
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer[name] = cv2.VideoWriter(os.path.join(self.save_dir, '%s.avi' % name),
                                                cv2.VideoWriter_fourcc(*'avc1'),
                                                self.framerate, (frame.shape[1], frame.shape[0]))
        self.writer[name].write(frame)

    def close(self):
        for wn in self.writer:
            self.writer[wn].release()
        self.writer.clear()


class GIFFileDisplayer(VisDisplayer):
    def __init__(self, save_dir=None, framerate=10, fig_size=None):
        if save_dir is None:
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.save_dir = os.path.join('.', 'vidout', current_time)
        else:
            self.save_dir = save_dir
        self.framerate, self.fig_size = framerate, fig_size
        self.figure = {}

    def display(self, name, frame, step=0):
        if name not in self.figure:
            os.makedirs(self.save_dir, exist_ok=True)
            self.figure[name] = []
        self.figure[name].append(frame)

    def close(self):
        for name, frames in self.figure.items():
            # bgr -> rgb
            frames = [frame[..., ::-1] for frame in frames]

            fig = plt.figure()
            fig.set_size_inches(np.array(frames[0].shape[1::-1]) / fig.dpi, forward=True)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

            ax = fig.gca()
            ax.axis('off')
            im = ax.imshow(frames[0], animated=True, aspect='equal')

            def _get_frame(i):
                im.set_array(frames[i + 1])
                return im,
            anim = FuncAnimation(fig, _get_frame, len(frames) - 1, blit=True)
            anim.save(os.path.join(self.save_dir, '%s.gif' % name), fps=self.framerate)
        self.figure.clear()


def matches(img1, pts1, img2, pts2, m_indices, colors, pts_colors, circ_radius=3, thickness=1, alpha=1):
    ''' Assume pts1 are matched with pts2, respectively.
    '''
    H1, W1, C = img1.shape
    H2, W2, _ = img2.shape
    new_img = np.zeros((max(H1, H2), W1 + W2, C), img1.dtype)
    new_img[:H1, :W1], new_img[:H2, W1:W1+W2] = img1,  img2
    overlay = new_img
    overlay = circles(overlay, pts1, circ_radius, pts_colors[0], thickness)
    pts2 = pts2.clone()
    pts2[:, 0] += W1
    overlay = circles(overlay, pts2, circ_radius, pts_colors[1], thickness)
    if m_indices is None:
        return overlay
    overlay =  lines(overlay, pts1[m_indices[:, 0]], pts2[m_indices[:, 1]], colors, thickness)
    return cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0)


def circles(image, points, radius, colors, thickness):
    for pt, c in zip(points, colors):
        if not torch.any(pt.isnan()):
            image = cv2.circle(image.copy(), tuple(pt.to(int).tolist()), radius, tuple(c.tolist()), thickness, cv2.LINE_AA)
    return image


def lines(image, pts1, pts2, colors, thickness):
    for pt1, pt2, c in zip(pts1, pts2, colors):
        if not torch.any(pt1.isnan() | pt2.isnan()):
            image = cv2.line(image.copy(), tuple(pt1.to(int).tolist()), tuple(pt2.to(int).tolist()), tuple(c.tolist()), thickness, cv2.LINE_AA)
    return image


def get_colors(name, values=None, vmin=None, vmax=None):
    values = np.array([0]) if values is None else values
    values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.array(values)
    if name in mc.get_named_colors_mapping():
        rgb = mc.to_rgba_array(name)[0, :3]
        rgb = np.broadcast_to(rgb, values.shape + (3,))
    else:
        normalize = mc.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(name)
        rgb = cmap(normalize(values))
    return (rgb[..., 2::-1] * 255).astype(np.uint8)


def torch2cv(images):
    if images.dtype != torch.uint8:
        images = (255 * images).type(torch.uint8)
    rgb = images.cpu().numpy()
    bgr = rgb[:, ::-1, ...].transpose((0, 2, 3, 1))
    return bgr
