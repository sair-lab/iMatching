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


data_url = 'https://www.eth3d.net/data/slam'

test_list = '/dataset_list_test.txt'
train_list = '/dataset_list_training.txt'

def call(f, *args, **kwargs):
    return f(*args, **kwargs)

def concat(a, b):
    return a + b

def endswith(s, b):
    return s.endswith(b)

def list_append(l, i):
    l.append(i)
    return l

def get(s, i, j=None, k=None, l=None):
    res = s[i]
    if j is not None:
        res = res[j]
    if k is not None:
        res = res[k]
    if l is not None:
        res = res[l]
    return res

def not_comment(s):
    return not s.startswith('#')

def string_replace(s, old, new):
    return s.replace(old, new)

def debug(s):
    print(s)
    return s

with_base_url = partial(os.path.join, data_url, 'datasets')


def scene_lister(*source):
    return OnlineReader(IterableWrapper(source)).readlines(return_path=False).map(bytes.decode)

def download_pipe(root: Union[str, Path], name_dp, suffix):
    url_dp = name_dp.map(with_base_url).map(partial(concat, b=suffix))
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=Compose([os.path.basename, partial(os.path.join, root)]) ,
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    # decompress
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=Compose([partial(str.split, sep=suffix), itemgetter(0), partial(concat, b='cache' + suffix)]),
        # for bookkeeping of the files extracted
    )
    cache_decompressed = cache_decompressed.open_files(mode="b").load_from_zip().end_caching(
        # filepath_fn=partial(get0, i=0)
        filepath_fn=Compose([partial(str.split, sep=suffix+os.sep), partial(get, i=-1), partial(os.path.join, root)])
    )
    return cache_decompressed

def associated_pose_demux(i):
    if i.endswith('associated.txt'):
        return 0
    elif i.endswith('groundtruth.txt'):
        return 1

def pose_reader(i):
    # tx ty tz qx qy qz qw
    pose = np.loadtxt(i, ndmin=2)
    timestamp = pose[:, 0]
    pose = pose[:, 1:]
    f = scipy.interpolate.interp1d(timestamp, pose, axis=0, fill_value="extrapolate")
    return f

def image_path(associated_path, img_path):
    return os.path.join(os.path.dirname(associated_path), img_path)

def build_pipeline(cache_path='test', split=train_list, has_pose=True):
    mono_pipe, rgbd_pipe = scene_lister(data_url + split).fork(2)
    mono_pipe = download_pipe(cache_path, mono_pipe, suffix='_mono.zip')
    rgbd_pipe = download_pipe(cache_path, rgbd_pipe, suffix='_rgbd.zip')
    combined = Concater(mono_pipe, rgbd_pipe)
    if has_pose:
        associated_pipe, pose_pipe = combined.demux(2, classifier_fn=associated_pose_demux, drop_none=True)
    else:
        associated_pipe = combined.filter(partial(endswith, b='associated.txt'))
        pose_pipe = None
    # parse associated.txt
    associated_pipe = associated_pipe.open_files(mode='r')\
        .readlines(return_path=True)\
        .map(' '.join)\
        .map(partial(str.split, sep=' '))\
        .slice([0, 1, 2, 4])\
        .map(float, input_col=1, output_col=1)
    # add K
    associated_pipe = associated_pipe\
        .map(partial(string_replace, old=os.sep+'associated.txt', new=os.sep+'calibration.txt'), input_col=0, output_col=-1)\
        .map(partial(np.loadtxt), input_col=-1)
    # path of images
    associated_pipe = associated_pipe.map(image_path, input_col=[0, 2], output_col=2)
    # path of depth
    associated_pipe = associated_pipe.map(image_path, input_col=[0, 3], output_col=3)
    # load pose if available
    if pose_pipe is not None:
        pose_pipe = pose_pipe.filter(partial(endswith, b='groundtruth.txt'))
        pose_pipe = pose_pipe.open_files(mode='r').map(pose_reader, input_col=1, output_col=1)
        pose_pipe = IterToMapConverter(pose_pipe, )
        associated_pipe = MapKeyZipper(associated_pipe, pose_pipe,
                                       key_fn=Compose([itemgetter(0),
                                                       partial(string_replace, old='associated.txt', new='groundtruth.txt')]),
                                       )
        associated_pipe = associated_pipe.map(list_append, input_col=[0, 1])\
            .map(itemgetter(0))\
            .map(call, input_col=[-1, 1])
    return associated_pipe



@dataclass
class eth3d_stereoMetaData(MetaDataBase):
    jpg_paths: List[str]
    depth_paths: List[str]
    poses: np.ndarray
    K: np.ndarray

SceneKeyType = Tuple[str]
Img_Depth_Pose_K = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@hydra_instantiable(name="eth3d_slam", group="datamodule/dataset")
class eth3d_slam(DatasetBase[SceneKeyType, Img_Depth_Pose_K, eth3d_stereoMetaData]):

    def __init__(self, root, catalog_dir=".dscatalog", needs_depth=True, quat_pose=False, split="all", loc_eval=False):
        dataset_path = os.path.join(root, "eth3d_slam")
        self.quat_pose = quat_pose
        if split == 'train':
            self.pipe = build_pipeline(cache_path=dataset_path, split=train_list, has_pose=True)
        elif split == 'test':
            self.pipe = build_pipeline(cache_path=dataset_path, split=test_list, has_pose=False)
        elif split == 'all':
            self.pipe = build_pipeline(cache_path=dataset_path, split=train_list, has_pose=True)
            self.pipe = self.pipe.concat(build_pipeline(cache_path=dataset_path, split=test_list, has_pose=False))

        
        super().__init__(Path(dataset_path), "eth3dslam_" + split, catalog_dir, groupby_series_idx=None if not loc_eval else 0)

        self.to_tensor = T.ToTensor()

    def _populate(self):
        # downloads
        scenes = dict()
        for item in self.pipe:
            if type(item[1]) == str:
                associated, jpg_path, depth_path, K, pose_q = item
            else:
                associated, time_stamp, jpg_path, depth_path, K = item
                pose_q = None
            scene = os.path.basename(os.path.dirname(associated))
            if scene not in scenes:
                scenes[scene] = []
            scenes[scene].append((jpg_path, depth_path, pose_q, K))
        
        for scene in scenes:
            jpg_paths = [i[0] for i in scenes[scene]]
            depth_paths = [i[1] for i in scenes[scene]]
            pose_q = [i[2] for i in scenes[scene]]
            K = [i[3] for i in scenes[scene]]
            size = len(scenes[scene])

            self.register_series(scene, eth3d_stereoMetaData(poses=pose_q, size=size, jpg_paths=jpg_paths, depth_paths=depth_paths, K=K))

    def getitem_impl(self, series: SceneKeyType, idx: int) -> Img_Depth_Pose_K:
        # image
        image = Image.open(self.series[series].jpg_paths[idx])
        image = image.convert('RGB')
        image = self.to_tensor(image)

        depth_path = self.series[series].depth_paths[idx]
        # A pixel's value must be divided by 5000 to obtain its depth
        depth = self.to_tensor(Image.open(depth_path)).float() / 5000.0
        pose = self.series[series].poses[idx]
        if pose is not None:
            pose_q = self.series[series].poses[idx]
            pose_q = torch.from_numpy(pose_q).to(torch.float)
        else:
            pose_q = torch.zeros(7, dtype=torch.float)
            pose_q[3] = 1
        if self.quat_pose:
            pose = pose_q
        else:
            pose = pose_vec2pose_mat(pose_q) # @ NED2EDN.T

        K = self.series[series].K[idx]
        K = build_intrinsic(*K)
        K = torch.from_numpy(K).to(torch.float)
        return image, depth, pose, K


if __name__ == '__main__':
    root = '/data/common'
    my_eth3d_stereo = eth3d_slam(root, split="test", catalog_dir='.dscatalog')


