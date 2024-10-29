import bz2
from dataclasses import dataclass
import itertools
import re
import pathlib
import pickle
from copy import copy
from typing import Any, Dict, Generic, Iterable, List, Optional, Set, Tuple, TypeVar
from typing_extensions import Self
import warnings

import numpy as np

from torch.utils.data import Dataset, Sampler, default_collate

from utils.config import hydra_config

data_ty = TypeVar('data_ty')
key_ty = TypeVar('key_ty')
meta_ty = TypeVar('meta_ty', bound="MetaDataBase")
data_w_key_idx_ty = Tuple[data_ty, Tuple[key_ty, int]]


@dataclass
class MetaDataBase:
    size: int


class DatasetBase(Dataset[data_w_key_idx_ty[key_ty, data_ty]], Generic[key_ty, data_ty, meta_ty]):
    __dataset_states__ = []
    __dataset_states_base__ = ["_series"]

    def __init__(self, root: str, name: str, catalog_dir: str = None, groupby_series_idx: Optional[int] = None):
        self.path_prefix = pathlib.Path(root)
        self.name = name
        self._series: Dict[key_ty, meta_ty] = {}
        self._series_visible: Set[key_ty] = set()
        self.groupby = groupby_series_idx

        # load state dict if available
        if catalog_dir is not None:
            success = False
            catalog_path = pathlib.Path(catalog_dir) / f"{name}.pbz2"
            if catalog_path.is_file():
                try:
                    with bz2.BZ2File(catalog_path, "rb") as f:
                        state_dict = pickle.load(f)
                        for k, v in state_dict.items():
                            setattr(self, k, v)
                    success = True
                    print(f"Loaded catalog {catalog_path}")
                except Exception as e:
                    warnings.warn(f"Catalog corrupted:\n{repr(e)}\nRe-building catalog.")

            if not success:
                self._populate()
                catalog_attr = self.__dataset_states_base__ + self.__dataset_states__
                state_dict = {attr: getattr(self, attr) for attr in catalog_attr}

                catalog_path.parent.mkdir(parents=True, exist_ok=True)
                with bz2.BZ2File(catalog_path, "wb") as f:
                    pickle.dump(state_dict, f)
                print(f"Saved catalog {catalog_path}")
        else:
            self._populate()
        self._series_visible.update(self._series.keys())

    @property
    def series_keys(self) -> List[key_ty]:
        return [k for k in self._series.keys() if k in self._series_visible]

    @property
    def dataloader_groups(self):
        if self.groupby == None:
            # default: each series is in the group by its own
            return {self.get_series_id(k): [k] for k in self.series_keys}
        elif isinstance(self.groupby, int):
            # int: group the series by part at the specified index
            series_groups: Dict[str, List[key_ty]] = {}
            for k in self.series_keys:
                series_groups.setdefault(k[self.groupby], []).append(k)
            return series_groups
        else:
            raise ValueError()

    @property
    def series(self) -> Dict[key_ty, meta_ty]:
        return {k: v for k, v in self._series.items() if k in self._series_visible}

    def _populate(self):
        raise NotImplementedError()

    def get_size(self, series: key_ty) -> int:
        return self.series[series].size

    def getitem_impl(self, series: key_ty, idx: int) -> data_ty:
        raise NotImplementedError()

    def get_series_id(self, series: key_ty):
        return "_".join(np.atleast_1d(series).tolist())

    def __getitem__(self, index: Tuple[key_ty, int]) -> data_w_key_idx_ty[data_ty, key_ty]:
        series, idx = index
        assert series in self.series_keys, f"No series named: {series}"
        assert 0 <= idx < self.get_size(series), f"Index out of bound for {series}: {idx}"
        item = self.getitem_impl(series, idx)
        return item, index

    def include_exclude(self, include: str = None, exclude: str = None) -> Self:
        incl_pattern = re.compile(include) if include is not None else None
        excl_pattern = re.compile(exclude) if exclude is not None else None
        subset = copy(self)
        subset._series_visible = copy(self._series_visible)
        for series in list(self.series_keys):
            seq_id = self.get_series_id(series)
            if (incl_pattern and incl_pattern.search(seq_id) is None) or \
                    (excl_pattern and excl_pattern.search(seq_id) is not None):
                subset._series_visible.remove(series)
        return subset

    def include_exclude_by_id(self, include: Iterable[key_ty] = None, exclude: Iterable[key_ty] = None) -> Self:
        subset = copy(self)
        if include is not None:
            subset._series_visible = self._series_visible.intersection(include)
        if exclude is not None:
            subset._series_visible = self._series_visible.difference(exclude)
        return subset


    def register_series(self, name: key_ty, series: meta_ty):
        self._series[name] = series

    def rand_split(self, ratio: Iterable[float], seed: int = 42, at_least_one=True):
        series_keys = self.series_keys
        total, ratio = len(series_keys), np.array(ratio)
        ratio = ratio / sum(ratio)
        if at_least_one:
            fixed = ratio <= 0
            n_insuff = 0
            while not fixed.all():
                # force insufficient split to have ratio 1/n and redistribute ratios among the rest
                insuff = (ratio > 0) & (ratio < 1 / total)
                n_insuff += insuff.sum()
                if not insuff.any():
                    break
                ratio[insuff] = 1 / total
                fixed |= insuff
                adjustable = ratio[~insuff & ~fixed]
                ratio[~insuff & ~fixed] = adjustable / sum(adjustable) * (1 - n_insuff / total)
        split_idx = np.cumsum(np.round(ratio / sum(ratio) * total), dtype=np.int32)[:-1]
        subsets: List[Self] = []
        for perm in np.split(np.random.default_rng(seed=seed).permutation(total), split_idx):
            included_series = [series_keys[i] for i in sorted(perm)]
            subsets.append(self.include_exclude_by_id(include=included_series))
        return subsets

    def __len__(self):
        return sum(self.get_size(series) for series in self.series_keys)
