import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms as T

import numpy as np

from utils.config import hydra_config, hydra_instantiable

from .augment import RandAug
from .datasets import *
from .datasets.base import DatasetBase

wrapper_fn_ty = Callable[[dataset_ty], wrapper_ty]


class Datasets(str, Enum):
    TartanAir = "tartanair"
    eth3d_stereo = "eth3d_stereo"
    KITTI = "kitti"
    eth3d_slam = "eth3d_slam"

    @classmethod
    def from_config(cls, config):
        for m in cls:
            if m in config._target_.lower():
                return m
        raise ValueError(f"Unknown dataset: {config._target_}")


@hydra_config(name="stage_config_default", group="datamodule/stage_config")
@dataclass
class DataModuleStageConfig:
    dataset: Any = MISSING
    batch_size: int = 1
    scale: float = 1.0
    pad: bool = False
    size: Tuple[int, int] = (480, 640)
    augmentation: Optional[str] = None
    include: Optional[str] = None
    exclude: Optional[str] = None
    wrapper: Optional[Any] = None


@hydra_instantiable(name="datamodule", group="datamodule",
                    override_group={"train_config": "stage_config", "validate_config": "stage_config",
                                    "test_config": "stage_config"},
                    _recursive_=False) # we don't want to all datasets to initialize yet, 
                    # otherwise the same dataset may be loaded multiple times
class DataModule(pl.LightningDataModule):

    def __init__(self,
                 train_config: DataModuleStageConfig = DataModuleStageConfig(),
                 validate_config: DataModuleStageConfig = DataModuleStageConfig(),
                 test_config: DataModuleStageConfig = DataModuleStageConfig(),
                 num_workers: int = 4,
                 split_ratio: Optional[List] = [0.8, 0.1, 0.1],
                 split_seed: Optional[int] = 41,
                 include: Optional[str] = None,
                 exclude: Optional[str] = None):
        super().__init__()
        self.per_stage_config = {"train": train_config, "validate": validate_config, "test": test_config}
        self.num_workers = num_workers
        self.needs_depth = True
        self.split_seed = split_seed
        self.split_ratio = split_ratio
        self.global_include = include
        self.global_exclude = exclude

        self.train_data: DatasetBase = None
        self.val_data: DatasetBase = None
        self.test_data: DatasetBase = None
        self.dataset_split_cache: Dict[DataModule.Datasets, List[dataset_ty]] = {}
        self._initialized_stages = set()
        self.per_stage_datasets: Dict[str, dataset_ty] = {}
        self.per_stage_wrappers: Dict[str, wrapper_fn_ty] = {}

    def get_stage_config(self, stage: str):
        stage = "validate" if stage == "sanity_check" else stage
        return self.per_stage_config[stage] if stage in self.per_stage_config \
            else self.per_stage_config["default"]
    
    def _setup_stage(self, stage: str):
        if stage not in self._initialized_stages:
            # instantiate datasets and wrappers
            cfg = self.get_stage_config(stage)
            # ds = Datasets.from_config(cfg.dataset)

            # # avoid repeated dataset loading
            # if ds not in self.dataset_split_cache:
            #     full_set = hydra.utils.instantiate(cfg.dataset)
            #     self.dataset_split_cache[ds] = full_set \
            #         .include_exclude(self.global_include, self.global_exclude) \
            #         .rand_split(self.split_ratio, self.split_seed)
            dataset_trimmed = hydra.utils.instantiate(cfg.dataset) \
                .include_exclude(self.global_include, self.global_exclude)
            # TODO hardcode
            idx = ["train", "validate", "test"].index(stage)
            if self.split_ratio is not None:
                if type(self.split_ratio[idx]) in (float, int) \
                        and self.split_ratio[idx] >= 0:
                    ratio = np.array(self.split_ratio).clip(min=0.)

                    at_least_one = True
                    if len(dataset_trimmed.series_keys) <= 1:
                        ratio = np.zeros_like(ratio)
                        ratio[0] = 1.
                        at_least_one = False
                        warnings.warn("Dataset has only one series. It will be used for training only.")

                    dataset_splits = dataset_trimmed.rand_split(ratio, self.split_seed, at_least_one)
                    dataset = dataset_splits[idx]
                elif type(self.split_ratio[idx]) == str:
                    seperator = '+'
                    if seperator in self.split_ratio[idx]:
                        series_keys = self.split_ratio[idx].split(seperator)
                    else:
                        series_keys = [self.split_ratio[idx]]
                    dataset = dataset_trimmed.include_exclude_by_id(include=series_keys)

            else:
                # do not split by ratio, dataset have configurable split (e.g. 7scenes)
                # instead use the newly instantiated dataset
                dataset = dataset_trimmed

            self.per_stage_datasets[stage] = dataset.include_exclude(cfg.include, cfg.exclude)
            self.per_stage_wrappers[stage] = hydra.utils.instantiate(cfg.wrapper)

            self._initialized_stages.add(stage)

    def setup(self, stage: str = None):
        if stage == "fit":
            # fit stage needs validate data loaders
            self._setup_stage("train")
            self._setup_stage("validate")
        elif stage is not None:
            self._setup_stage(stage)
        else:
            for stage in ("train", "validate", "test"):
                self._setup_stage(stage)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        cfg = self.get_stage_config(self.trainer.state.stage)
        dataset_name = Datasets.from_config(cfg.dataset)

        if dataset_name in (Datasets.TartanAir, 
                            Datasets.eth3d_slam, Datasets.eth3d_stereo, Datasets.KITTI,):
            (images, depths, poses, Ks), aux = batch
            if not cfg.dataset.needs_depth:
                depths = [[None] * depths.shape[1]] * depths.shape[0]
            if self.trainer.training and cfg.augmentation == "all":
                augment = RandAug(cfg.scale, size=cfg.size, p=[0.3, 0.0, 0.3, 0.3], pad=cfg.pad)
            elif cfg.augmentation is None:
                augment = RandAug(cfg.scale, size=cfg.size, p=[1.0, 0.0, 0.0, 0.0], pad=cfg.pad)
            images_aug, Ks_aug, depths_aug, poses_aug = for_each_tensor_tuple_apply(
                partial(for_each_tensor_tuple_apply, augment),
                images, Ks, depths, poses)
            return (images_aug, depths_aug, poses_aug, Ks_aug), aux
        elif dataset_name == Datasets.HPatches:
            return batch
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def train_dataloader(self):
        return self._make_loaders_for_stage("train", split_by_group=False)

    def val_dataloader(self):
        return self._make_loaders_for_stage("validate", split_by_group=True)

    def test_dataloader(self):
        return self._make_loaders_for_stage("test", split_by_group=True)

    def _make_loaders_for_stage(self, stage: str, split_by_group=False):
        cfg = self.get_stage_config(stage)
        data = self.per_stage_datasets[stage].include_exclude(cfg.include, cfg.exclude)
        wrapper_fn = self.per_stage_wrappers[stage]

        if not split_by_group:
            wrapped_partial_set = wrapper_fn(data)
            return self._make_loader(wrapped_partial_set, wrapped_partial_set.collate_fn, cfg)
        else:
            data_loaders: List[wrapper_ty] = []
            for _, series_keys in data.dataloader_groups.items():
                wrapped_partial_set = wrapper_fn(data.include_exclude_by_id(include=series_keys))
                loader = self._make_loader(wrapped_partial_set, wrapped_partial_set.collate_fn, cfg)
                if len(loader) > 0:
                    data_loaders.append(loader)
                else:
                    warnings.warn(f"Series {series_keys} has length 0. Skipped.")
            return data_loaders

    def _make_loader(self, dataset: Dataset, collate_fn, cfg: DataModuleStageConfig):
        return DataLoader(dataset, batch_size=cfg.batch_size,
                          pin_memory=True, num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers > 0 else False, collate_fn=collate_fn)


def for_each_tensor_tuple_apply(func: Callable[..., Tuple[torch.Tensor, ...]], *tensors: torch.Tensor):
    # [func(x00, ..., xn0), func(x01, ..., xn1), ...]
    transformed = (func(*args) for args in zip(*tensors))
    return (torch.stack(tensors_transf) if isinstance(tensors_transf[0], torch.Tensor) else tensors_transf
            for tensors_transf in zip(*transformed))
