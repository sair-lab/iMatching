from __future__ import annotations

import sys
from abc import ABC
from copy import copy
from dataclasses import dataclass, field, fields
from itertools import chain
from typing import Any, ClassVar, Dict, Generic, Iterable, List, MutableSequence, Sequence, Tuple, Type, TypeVar, Union

from typing_extensions import Self

IT = TypeVar('IT', bound='_IndexMixin')
OT = TypeVar('OT')
Corrs = FeaId = Tuple[int, int]
GlobalCorrs = Tuple[FeaId, FeaId]
FeaIdList = List[FeaId]
CorrsTable = Dict[int, List[Corrs]]
FeaIdTable = Dict[int, List[FeaId]]
INVALID_INDEX = sys.maxsize


class _IndexedStore(Sequence[IT], Generic[IT, OT], ABC):
    def __init__(self, item_cls: Type[IT], owner: OT, shared_values: Dict[str, Any] = None) -> None:
        # use list as default storage type
        self.field_names: List[str] = item_cls.get_data_field_names()
        self._master = self
        self._cum_idx = 0
        self.item_cls = item_cls
        self._default_item = item_cls(_store=self)
        self.owner = owner
        self.shared_values = {} if shared_values is None else shared_values

    @property
    def iloc(self):
        return _ILocAccessor(self)

    def add(self, single=False, fill_default=True, **values):
        if self._master != self:
            raise ValueError('Cannot add to view.')

        if set(values.keys()) != set(self.field_names) and not fill_default:
            raise ValueError(f'Missing or extra keys: {set(values.keys())} (expected: {set(self.field_names)}).')

        add_size = -1
        if not single:
            sizes = [len(v) for v in values.values() if v is not None]
            if len(sizes) < 1:
                raise ValueError('Cannot determine number of items added.')
            elif not all([size == sizes[0] for size in sizes]):
                raise ValueError(f'Inconsistent sizes: {sizes}.')
            add_size = sizes[0]
        else:
            add_size = 1

        data_dict = {}
        # init vars do not appear in fields but should be included in data_dict to construct object
        init_vars = [k for k, v in self.item_cls.__annotations__.items() if v.startswith("InitVar")]
        for field in chain(self.field_names, init_vars):
            is_default = False
            if fill_default and field not in values:
                value = copy(getattr(self._default_item, field))
                is_default = True
            else:
                value = values[field]

            # wrap single value or duplicate default value, which is single
            if single:
                value = [value]
            elif value is None or is_default:
                value = [copy(value) for _ in range(add_size)]

            data_dict[field] = value

        data_dict['_idx'] = self._allocate_new_idx(add_size)
        self._add_impl(data_dict)

    def remove(self, item: IT):
        item._assert_in_storage()
        if self._master != self:
            raise ValueError('Cannot remove from view.')
        self._remove_impl(item)
        item._idx = INVALID_INDEX

    def _allocate_new_idx(self, size: int) -> Iterable[int]:
        ret = range(self._cum_idx, self._cum_idx + size)
        self._cum_idx += size
        return ret

    def _add_impl(self, data_dict: Dict[str, MutableSequence]):
        pass

    def _remove_impl(self, item: IT):
        pass

    def item_setittr(self, item: IT, name: str, value: Any):
        pass

    def get_iloc(self, index) -> int:
        pass

    def _iloc_impl(self, index: Union[int, slice]) -> Union[IT, Self]:
        pass

    def _getitem_impl(self, index: int) -> IT:
        pass

    def __len__(self):
        pass

    def __getitem__(self, index: int) -> IT:
        # syntactic sugar: negative index to get the latest frames
        if index < 0:
            return self._iloc_impl(len(self) + index)
        else:
            return self._getitem_impl(index)

    def __iter__(self):
        return self.iloc.__iter__()


@dataclass
class _IndexMixin(Generic[IT, OT]):
    _field_names: ClassVar[List[str]] = None

    _idx: int = INVALID_INDEX
    _store: _IndexedStore[IT, OT] = field(default=None, repr=False)
    _attr_dispatch: bool = field(default=False, repr=False)

    def __post_init__(self):
        # only activate __setattr__ after full init
        self._attr_dispatch = True

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def iloc(self) -> int:
        return self._store.get_iloc(self.idx)

    @property
    def in_storage(self):
        return self._idx != INVALID_INDEX

    @property
    def owner(self) -> OT:
        return self._store.owner

    @classmethod
    def get_data_field_names(self) -> List[str]:
        return self._field_names

    def __setattr__(self, name: str, value: Any):
        # write back to storage
        if self._attr_dispatch and self.in_storage and name in self._field_names:
            self._store.item_setittr(self, name, value)
        super().__setattr__(name, value)

    def _assert_in_storage(self):
        if not self.in_storage:
            raise ValueError('Item does not belong to a storage.')

    # def __eq__(self, other) -> bool:
    #     return isinstance(other, self.__class__) and self._idx == other._idx

    def __hash__(self) -> int:
        return hash((self.__class__, self._idx))


class _ILocAccessor(Sequence[IT], Generic[IT, OT], ABC):
    def __init__(self, store: _IndexedStore[IT, OT]) -> None:
        self._store = store

    def __getitem__(self, index) -> Union[IT, _IndexedStore[IT, OT]]:
        return self._store._iloc_impl(index)

    def __len__(self) -> int:
        return len(self._store)


def get_unique(items: Iterable[IT]):
    recorded_id = set()
    for item in items:
        if item.idx not in recorded_id:
            yield item
        recorded_id.add(item.idx)


def indexed_dataclass(cls: Type[IT] = None, field_exclude: List[str] = [], *args, **kwargs):
    def wrapper(cls: Type[IT]):
        cls = dataclass(cls, *args, **kwargs)
        # populate class attributes
        _IndexMixin_names = [f.name for f in fields(_IndexMixin)]
        cls._field_names = [
            f.name for f in fields(cls) if f.name not in _IndexMixin_names and f.name not in field_exclude
        ]
        return cls
    return wrapper if cls is None else wrapper(cls)
