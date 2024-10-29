from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
import tqdm

from utils.config import hydra_instantiable

from dataset.datasets.base import DatasetBase, MetaDataBase


SceneKeyType = Tuple[str]
Img_Depth_Pose_K = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# eth3d_stereo
from functools import partial
from typing import Union
import numpy as np
import py7zr
import torchdata
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper, \
    IterDataPipe, Zipper, IterKeyZipper
from torchvision.transforms import Compose

train_scenes = [
    # training scenes
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",]
test_scenes = [
    "botanical_garden",
    "boulders",
    "bridge",
    "door",
    "exhibition_hall",
    "lecture_room",
    "living_room",
    "lounge",
    "observatory",
    "old_computer",
    "statue",
    "terrace_2", ]
full_scenes = train_scenes + test_scenes
base_url = 'https://www.eth3d.net/data/'



class Decompressor7z(IterDataPipe):
    def __init__(self, dp) -> None:
        self.dp = dp

    def __iter__(self):
        for file in self.dp:
            with py7zr.SevenZipFile(file, 'r') as zip:
                yield from zip.readall().items()  # key: filename


def download_pipe(root: Union[str, Path], scenes):
    root = os.fspath(root)
    # download
    url_dp = IterableWrapper([archive_name for archive_name in scenes])
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=Compose([os.path.basename, partial(os.path.join, root)]) ,
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    # decompress
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=Compose([partial(str.split, sep='.'), list.pop, partial(os.path.join, root)])
        # for book keeping of the files extracted
    )
    cache_decompressed = Decompressor7z(cache_decompressed).end_caching(
        filepath_fn=partial(os.path.join, root)
    )
    return cache_decompressed


def build_intrinsic(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def demux_func(file):
    if 'cameras.txt' in file:
        return 0
    elif 'points3D.txt' in file:
        return 1
    elif 'images.txt' in file:
        return 2
    elif '.JPG' in file:
        return 3
    else:
        return None


def load_camera(file):
    # skip first three lines of comments occur in all files
    camera_ids = np.loadtxt(file, skiprows=3, usecols=(0,), dtype=int, ndmin=1) #CAMERA_ID
    camera_data = np.loadtxt(file, skiprows=3, usecols=(2, 3, 4, 5, 6, 7),
                             ndmin=2)  # WIDTH HEIGHT PARAMS[fx, fy, cx, cy,]
    camera_dict = {k.item(): build_intrinsic(*(v[2:])) for k, v in
                   zip(camera_ids, camera_data)}
    return camera_dict


def load_points(file):
    # each row: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    point3d_ids = np.loadtxt(file, usecols=(0,), dtype=int, ndmin=1)  # (N,)
    point3d_xyz = np.loadtxt(file, usecols=(1, 2, 3), dtype=np.float32, ndmin=2)  # (N, 3)
    point3d_rgb = np.loadtxt(file, usecols=(4, 5, 6), dtype=int, ndmin=2)  # (N, 3)

    point3d_dict = {k.item(): (xyz, rgb) for k, xyz, rgb in
                    zip(point3d_ids, point3d_xyz, point3d_rgb)}
    return point3d_dict


def colmap2lietensor(x: np.array):
    # colmap QW QX QY QZ TX TY TZ
    import pypose as pp
    return pp.SE3(x[..., [4, 5, 6, 1, 2, 3, 0]])


def colmap2tartan(x: np.array):
    # colmap QW QX QY QZ TX TY TZ
    # tartan tx, ty, tz, qx, qy, qz, qw]
    return x[..., [4, 5, 6, 1, 2, 3, 0]]


def parse_image(data):
    ((camera, point, filename), first_line), (_, second_line) = data
    # first line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    pose = np.fromstring(first_line, dtype=np.float32, sep=' ', count=8)[1:]  # (7,)
    first_line = first_line.split()
    image_id = int(first_line[0])
    camera_id = int(first_line[8])
    jpg_name = first_line[9]

    # second line: POINTS2D[] as (X, Y, POINT3D_ID)
    pixels = np.fromstring(second_line, dtype=np.float32, sep=' ').reshape(-1, 3)[:, :2]
    point_ids = np.array([int(i) for i in second_line.split()[2::3]])
    assert len(point_ids) == len(pixels)
    return dict(image_txt=filename, jpg_name=jpg_name, image_id=image_id,
                camera_id=camera_id, pixels=pixels, point_ids=point_ids,
                pose=colmap2tartan(pose), camera=camera, point=point, )


def get(stroage, item):
    return stroage[item]

def merge_jpg_name(x, y): 
    return x | {'jpg_path': y}


def load_image(cache_pipe):
    camera, point, image, jpg = cache_pipe.demux(4, demux_func, drop_none=True)
    image_file, image_io = FileOpener(image).unzip(2)
    annotation = Zipper(camera.map(load_camera), point.map(load_points), image_file)
    image = Zipper(annotation, image_io)
    image = image.readlines(skip_lines=4).batch(2).map(parse_image)

    return IterKeyZipper(image, jpg,
                        key_fn=Compose([partial(get, item='jpg_name'), os.path.basename]),
                        ref_key_fn=os.path.basename,
                        merge_fn=merge_jpg_name,
                        keep_key=False)

@dataclass
class eth3d_stereoMetaData(MetaDataBase):
    jpg_paths: List[str]
    depth_paths: List[str]
    poses: np.ndarray
    K: np.ndarray



@hydra_instantiable(name="eth3d_stereo", group="datamodule/dataset")
class eth3d_stereo(DatasetBase[SceneKeyType, Img_Depth_Pose_K, eth3d_stereoMetaData]):

    def __init__(self, root, catalog_dir="./.dscatalog", needs_depth=False, split="train", loc_eval=False):
        if needs_depth:
            if split == 'train':
                depth_archives = [base_url + i + '_dslr_depth.7z' for i in train_scenes]
            else:
                raise ValueError("eth3d_stereo does not provide depth for test scenes")
        else:
            depth_archives = None

        if split == 'train':
            img_archives = [base_url + i + '_dslr_undistorted.7z' for i in train_scenes]
        elif split == 'test':
            img_archives = [base_url + i + '_dslr_undistorted.7z' for i in test_scenes]
        else:
            img_archives = [base_url + i + '_dslr_undistorted.7z' for i in full_scenes]
        dataset_path = os.path.join(root, "eth3d")
        self.img_pipe = load_image(download_pipe(dataset_path, img_archives))
        self.depth_pipe = download_pipe(dataset_path, depth_archives) if depth_archives else None
        
        super().__init__(Path(dataset_path), "eth3d", catalog_dir, groupby_series_idx=None if not loc_eval else 0)

        self.needs_depth = needs_depth
        self.to_tensor = T.ToTensor()

    def _populate(self):
        # downloads
        for _ in tqdm.tqdm(self.img_pipe, desc="Downloading images"):
            pass
        if self.depth_pipe is not None:
            for _ in tqdm.tqdm(self.depth_pipe, desc="Downloading depth"):
                pass
        
        scenes = dict()
        for item in self.img_pipe:
            pose_q = item['pose']
            scene: str = item['image_txt'].split('/dslr_calibration_undistorted/images.txt')[0].split('/')[-1]
            jpg_path = item['jpg_path']
            depth_path = jpg_path.replace('images/dslr_images_undistorted', 'ground_truth_depth/dslr_images')
            K = item['camera'][item['camera_id']]
            if scene not in scenes:
                scenes[scene] = list()
            scenes[scene].append((jpg_path, depth_path, pose_q, K))
        
        for scene in scenes:
            scenes[scene] = sorted(scenes[scene], key=partial(get, item=0))
            jpg_paths = [i[0] for i in scenes[scene]]
            depth_paths = [i[1] for i in scenes[scene]]
            pose_q = np.stack([i[2] for i in scenes[scene]])
            K = [i[3] for i in scenes[scene]]
            size = len(scenes[scene])

            self.register_series(scene, eth3d_stereoMetaData(poses=pose_q, size=size, jpg_paths=jpg_paths, depth_paths=depth_paths, K=K))

    def getitem_impl(self, series: SceneKeyType, idx: int) -> Img_Depth_Pose_K:
        image = self.to_tensor(Image.open(self.series[series].jpg_paths[idx]))
        if self.needs_depth:
            depth_path = self.series[series].depth_paths[idx]
            # open as binary and load as numpy array
            with open(depth_path, "rb") as f:
                depth = np.fromfile(f, dtype=np.float32)
            depth = self.to_tensor(depth[None])
            depth = depth.reshape(1, *image.shape[-2:])
        else:
            depth = torch.empty(0, *image.shape[-2:])

        pose, K = self.series[series].poses[idx], self.series[series].K[idx]

        return image, depth, torch.from_numpy(pose).to(torch.float), torch.from_numpy(K).to(torch.float)


if __name__ == '__main__':
    root = '/data/common/eth3d'
    my_eth3d_stereo = eth3d_stereo(root, split="train", needs_depth=False)
    my_eth3d_stereo['delivery_area', 0]