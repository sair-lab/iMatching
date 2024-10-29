import hashlib
import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Generic, List, Optional, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm

from utils.config import hydra_instantiable

from ..augment import RandAug
from .base import DatasetBase, data_ty, data_w_key_idx_ty, key_ty, meta_ty
import networkx as nx
from itertools import combinations

def transpose(x):
    return x.swapdims(0, 1) if isinstance(x, torch.torch.Tensor) else list(zip(*x))


def collate_batched_seq(batch, data_collate=default_collate):
    data, info = transpose(batch)
    # n: length of series key (e.g. 3 for tartanair)
    series_keys_bxn, indices = transpose(info)
    series_keys_nxb = tuple(transpose(series_keys_bxn))
    return data_collate(data), (series_keys_nxb, default_collate(indices))


def collate_batched_seq_no_stack(batch):
    # only batch tensors by location in output tuple but don't stack yet:
    # dimensions may not match for v_* sequences
    return collate_batched_seq(batch, data_collate=lambda x: transpose(x))


class CollateFunc(Enum):
    default = auto()
    batched_seq = auto()
    batched_seq_no_stack = auto()


class _CollateFunctionMixin:

    _funcs = {
        CollateFunc.default: default_collate,
        CollateFunc.batched_seq: collate_batched_seq,
        CollateFunc.batched_seq_no_stack: collate_batched_seq_no_stack,
    }

    def __init__(self, collate_fn: CollateFunc = CollateFunc) -> None:
        # if collate_fn not in self._funcs:
        #     names = list(self._funcs.keys())
        #     raise ValueError(f"Unknown collate function: {collate_fn}. Available options are: {names}")
        self._collate_fn = self._funcs[collate_fn]

    @property
    def collate_fn(self):
        return self._collate_fn


@dataclass
class DatasetWrapperConfig:
    collate_fn: CollateFunc = CollateFunc.default


# setting _partial_=True will generate a partial function at instantiation
# https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation
@hydra_instantiable("no-wrapper", "datamodule/wrapper", _partial_=True)
class Nowrapper(
        Dataset[data_w_key_idx_ty[data_ty, key_ty]],
        Generic[key_ty, data_ty, meta_ty],
        _CollateFunctionMixin):

    def __init__(self, dataset: DatasetBase[key_ty, data_ty, meta_ty]) -> None:
        super().__init__(collate_fn=CollateFunc.default)
        self.dataset = dataset

    def __getitem__(self, index: key_ty):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


@dataclass
class SpacedSequenceConfig(DatasetWrapperConfig):
    collate_fn: CollateFunc = CollateFunc.batched_seq
    seq_len: int = 1
    seq_spacing: int = 1
    item_spacing: int = 1
    shuffle_seq: bool = False
    shuffle_series_within: Optional[str] = None


@hydra_instantiable(name="spaced-sequence", group="datamodule/wrapper", _partial_=True)
class SpacedSequence(
        Dataset[data_w_key_idx_ty[data_ty, key_ty]],
        Generic[key_ty, data_ty, meta_ty],
        _CollateFunctionMixin):

    def __init__(self, dataset: DatasetBase[key_ty, data_ty, meta_ty], cfg: SpacedSequenceConfig = SpacedSequenceConfig()):
        super().__init__(collate_fn=cfg.collate_fn)
        self.dataset = dataset
        self.batch_info = self.gen_sequence_batch(dataset, cfg.seq_len, cfg.seq_spacing, cfg.item_spacing,
                                             cfg.shuffle_seq, cfg.shuffle_series_within)

    def __getitem__(self, index: int) -> data_w_key_idx_ty[data_ty, key_ty]:
        series, indices = self.batch_info[index]
        return default_collate([self.dataset[(series, idx)] for idx in indices])

    def __len__(self):
        return len(self.batch_info)

    @staticmethod
    def gen_sequence_batch(
            dataset: DatasetBase[key_ty, data_ty, meta_ty],
            seq_len: int, seq_spacing=1, item_spacing=1, shuffle_seq=False, shuffle_series_within=None):
        assert seq_spacing > 0 and item_spacing > 0
        batch_info: List[Tuple[key_ty, List[int]]] = []

        for series_id in dataset.series_keys:
            size = dataset.get_size(series_id)
            n_batch = (size - ((seq_len - 1) * item_spacing + 1)) // seq_spacing + 1
            batch_idx = np.lib.stride_tricks.as_strided(
                np.arange(0, size), [n_batch, seq_len], [seq_spacing * 8, item_spacing * 8], writeable=False)
            if shuffle_seq:
                # (0 1 2) (3 4 5) -> (2 0 1) (3 5 4)
                np.random.shuffle(batch_idx.T)
            batch_info.extend((series_id, idx) for idx in batch_idx)

        return shuffle_items(batch_info, shuffle_series_within)


HPatchesPair_TY = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        Tuple[Tuple[Tuple[str], Tuple[str]], Tuple[int]]]



def shuffle_items(batch_info: List[Tuple[key_ty, List[int]]], shuffle_series_within: str = None):
    if shuffle_series_within is None:
        # A1 A2 A3 B1 B2 B3
        return batch_info

    if shuffle_series_within == 'env':
        batch_info_by_env: Dict[Tuple, List[Tuple[key_ty, List[List[int]]]]] = {}
        for batch_info_ in batch_info:
            series_id, batch_idx = batch_info_
            batch_info_by_env.setdefault(series_id[0], []).append((series_id, batch_idx))
        # A1 A3 A2 B3 B1 B2
        def _shuffle(x):
            np.random.shuffle(x)
            return x
        batch_info = list(itertools.chain(*[_shuffle(seq) for seq in batch_info_by_env.values()]))
    elif shuffle_series_within == 'all':
        # B1 A2 B2 A1 A3 B3
        batch_info = list(batch_info)
        np.random.shuffle(batch_info)
    
    return batch_info
