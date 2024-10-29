from copy import copy
from typing import Any, Dict, List, MutableSequence, Type

from .indexd_store import _IndexedStore, IT, OT


class IndexedStore(_IndexedStore[IT, OT]):
    def __init__(self, item_cls: Type[IT], owner: OT, shared_values: Dict[str, Any] = None) -> None:
        super().__init__(item_cls, owner, shared_values)
        # use list as default storage type
        self._store: Dict[int, IT] = {}
        self._index: List[int] = []

    def _add_impl(self, data_dict: Dict[str, MutableSequence]):
        for values in zip(*data_dict.values()):
            kwargs = {k: v for k, v in zip(data_dict.keys(), values)}
            kwargs['_store'] = self._master
            kwargs.update(self.shared_values)
            new_item = self.item_cls(**kwargs)
            self._store[new_item.idx] = new_item
            self._index.append(new_item.idx)

    def _remove_impl(self, item: IT):
        self._store.pop(item.idx)
        self._index.remove(item.idx)

    def item_setittr(self, item: IT, name: str, value: Any):
        # prevent recursion
        _dispatch = item._attr_dispatch
        item._attr_dispatch = False
        setattr(item, name, value)
        item._attr_dispatch = _dispatch

    def get_iloc(self, index) -> int:
        if isinstance(index, int):
            return self._index.index(index)
        else:
            raise ValueError(f'Invalid index: {index}')

    def _iloc_impl(self, index):
        if isinstance(index, int):
            return self.__getitem__(self._index[index])
        elif isinstance(index, slice):
            new_store = copy(self)
            new_store._index = self._index[index]
            new_store._store = {i: self._store[i] for i in new_store._index}
            return new_store
        else:
            raise ValueError(f'Invalid index: {index}')

    def _getitem_impl(self, index):
        return self._store[index]

    def __len__(self):
        return len(self._index)
