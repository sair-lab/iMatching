from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from dataset.datasets.eth3d import build_intrinsic
from dataset.datasets.tartanair import NED2EDN

from utils.config import hydra_instantiable

from dataset.datasets.base import DatasetBase, MetaDataBase

import os
import sys
from functools import partial
from operator import itemgetter, methodcaller, contains
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import scipy
import torchdata
import tqdm
from torch.nn import Identity
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper, \
    IterDataPipe, Zipper, IterKeyZipper, OnlineReader, Decompressor, Concater, \
    MapKeyZipper
from torchdata.datapipes.map import IterToMapConverter
from torchvision.transforms import Compose

from utils.geometry import pose_vec2pose_mat


train_list = [
    '2013_05_28_drive_0000_sync',
    '2013_05_28_drive_0003_sync',
    '2013_05_28_drive_0005_sync',
    '2013_05_28_drive_0007_sync',
    '2013_05_28_drive_0010_sync',
    '2013_05_28_drive_0002_sync',
    '2013_05_28_drive_0004_sync',
    '2013_05_28_drive_0006_sync',
    '2013_05_28_drive_0009_sync',
]

@dataclass
class kitti_MetaData(MetaDataBase):
    imgs: List[str]
    poses: np.ndarray
    K: np.ndarray

SceneKeyType = Tuple[str]
Img_Depth_Pose_K = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@hydra_instantiable(name="kitti", group="datamodule/dataset")
class KITTI(DatasetBase[SceneKeyType, Img_Depth_Pose_K, kitti_MetaData]):

    def __init__(self, root, catalog_dir=".dscatalog", quat_pose=False, split="all", needs_depth=False, loc_eval=False):
        dataset_path = os.path.join(root, "kitti360")
        self.quat_pose = quat_pose
        if split == "all":
            self.seq_list = train_list
        
        super().__init__(Path(dataset_path), "kitti360_" + split, catalog_dir, groupby_series_idx=None if not loc_eval else 0)

        self.to_tensor = T.ToTensor()

    def _populate(self):
        for seq in self.seq_list:
            imgs = []
            img_path = self.path_prefix / 'data_2d_raw' / seq / 'image_00' / 'data_rect'
            pos_path = self.path_prefix / 'data_poses' / seq / 'cam0_to_world.txt'
            itc_path = self.path_prefix / 'calibration' / 'perspective.txt'
            if not img_path.is_dir():
                raise RuntimeError(f"Please download the dataset and extract it to {self.path_prefix}")
            assert pos_path.is_file(), f"Please download the dataset and extract it to {self.path_prefix}"
            # for image in img_path.glob("*.png"):
            #     imgs.append(image)
            # imgs.sort()
            poses = np.loadtxt(pos_path, dtype=np.float32, delimiter=' ', usecols=range(1, 1+16))
            time_stamps = np.loadtxt(pos_path, dtype=int, delimiter=' ', usecols=0)
            # fill images in 10 digit format
            mask = np.zeros_like(time_stamps, dtype=bool)
            for idx, time_stamp in enumerate(time_stamps):
                if (img_path / f'{time_stamp:010d}.png').is_file():
                    imgs.append(img_path / f'{time_stamp:010d}.png')
                    mask[idx] = True
            # find intrinsics
            with open(itc_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('P_rect_00'):
                        nums = line.rstrip('\n').lstrip('P_rect_00: ').split(' ')
                        nums = [float(num) for num in nums]
                        K = np.array(nums).reshape(3, 4)
                        break
            poses = poses[mask]
            time_stamps = time_stamps[mask]
            assert poses.shape[0] == len(imgs) == len(time_stamps)
            size = len(poses)

            self.register_series(seq, kitti_MetaData(poses=poses, size=size, imgs=imgs, K=K))

    def getitem_impl(self, series: SceneKeyType, idx: int) -> Img_Depth_Pose_K:
        # image
        image = Image.open(self.series[series].imgs[idx])
        image = image.convert('RGB')
        image = self.to_tensor(image)

        depth = torch.zeros_like(image[[0,]])
        pose = self.series[series].poses[idx].reshape(4,4)

        K = self.series[series].K[:3, :3]
        K = torch.from_numpy(K).to(torch.float)
        return image, depth, pose, K
