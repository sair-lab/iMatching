import os
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

from utils.config import hydra_instantiable
from utils.geometry import pose_vec2pose_mat

from .base import DatasetBase, MetaDataBase


@dataclass
class TartanAirMetaData(MetaDataBase):
    pose: np.ndarray


Env_Diff_Pxxx = Tuple[str, str, str]
Img_Depth_Pose_K = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

NED2EDN = torch.FloatTensor([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

@hydra_instantiable(name="tartanair", group="datamodule/dataset")
class TartanAir(DatasetBase[Env_Diff_Pxxx, Img_Depth_Pose_K, TartanAirMetaData]):

    def __init__(self, root, catalog_dir='./.dscatalog', quat_pose=False, needs_depth=False):
        self.quat_pose = quat_pose
        super().__init__(pathlib.Path(root) / "tartanair", "tartanair", catalog_dir)
        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = 320., 320., 320., 240.
        self.K = torch.Tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        self.needs_depth = needs_depth
        self.to_tensor = T.ToTensor()

    def _populate(self):
        for env_path in sorted(self.path_prefix.glob('*')):
            env = env_path.stem
            for p in sorted(env_path.glob('[EH]a[sr][yd]/*')):
                dif, id = p.parts[-2:]
                seq_path = self.path_prefix / env / dif / id
                pose_q = np.loadtxt(seq_path / 'pose_left.txt', dtype=np.float32)
                size = len([p for p in os.listdir(seq_path / 'image_left/') if p.endswith('.png')])
                self.register_series((env, dif, id), TartanAirMetaData(pose=pose_q, size=size))

    def getitem_impl(self, series, idx):
        env, dif, id = series
        seq_path = self.path_prefix / env / dif / id
        image = self.to_tensor(Image.open(seq_path / 'image_left' / ('%0.6d_left.png' % idx)))

        if self.needs_depth:
            depth = F.to_tensor(np.load(seq_path / 'depth_left' / ('%0.6d_left_depth.npy' % idx)))
        else:
            depth = torch.empty(0, *image.shape[-2:])

        pose_q: torch.Tensor = self.series[series].pose[idx]
        pose = pose_q if self.quat_pose else pose_vec2pose_mat(pose_q) @ NED2EDN.T

        return image, depth, pose, self.K

    # def summary(self):
    #     return pd.DataFrame(data=[
    #         list(series) + [self.get_size(series)] for series in self.series_keys()],
    #         columns=['env', 'dif', 'id', 'size'])


MExxx = Tuple[str]
Img_Pose_K = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@hydra_instantiable(name="tartanair_test", group="datamodule/dataset")
class TartanAirTest(DatasetBase[MExxx, Img_Pose_K, TartanAirMetaData]):
    NED2EDN = torch.FloatTensor([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    def __init__(self, root, catalog_dir='./.dscatalog', quat_pose=False):
        self.quat_pose = quat_pose
        super().__init__(pathlib.Path(root) / "tartanair_test", "tartanair_test", catalog_dir)
        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = 320., 320., 320., 240.
        self.K = torch.Tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        self.to_tensor = T.ToTensor()

    def _populate(self):
        for seq_path in sorted((self.path_prefix / "mono").glob('*')):
            seq = seq_path.stem
            pose_q = np.loadtxt(self.path_prefix / "mono_gt" / f"{seq}.txt", dtype=np.float32)
            size = len([p for p in os.listdir(seq_path) if p.endswith('.png')])
            self.register_series((seq,), TartanAirMetaData(pose=pose_q, size=size))

    def getitem_impl(self, series, idx):
        image = self.to_tensor(Image.open(self.path_prefix / "mono" / series[0] / ('%0.6d.png' % idx)))

        pose_q: torch.Tensor = self.series[series].pose[idx]
        pose = pose_q if self.quat_pose else pose_vec2pose_mat(pose_q) @ NED2EDN.T

        return image, pose, self.K
