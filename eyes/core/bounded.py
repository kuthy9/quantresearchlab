"""An insertion-bounded dictionary for the bar-keyed memories.

The emitter remembers, per bar root, its close and price range; the store
its same-clock reservations.  Every such memory grew for the life of the
run although each is read within a clock window fixed when the reader
forms.  The container keeps the newest ``maxlen`` entries and evicts the
oldest insertion; a lookup of an evicted key answers exactly as a
never-seen key does, and every reader of these memories fails closed on a
miss.
"""
from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedDict(MutableMapping, Generic[K, V]):
    """A dictionary that keeps its newest ``maxlen`` insertions."""

    __slots__ = ("_items", "maxlen")

    def __init__(self, maxlen: int) -> None:
        if type(maxlen) is not int or maxlen < 1:
            raise ValueError("bounded dictionary needs a positive maxlen")
        self.maxlen = maxlen
        self._items: dict[K, V] = {}

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __setitem__(self, key: K, value: V) -> None:
        items = self._items
        if key in items:
            items[key] = value
            return
        if len(items) >= self.maxlen:
            del items[next(iter(items))]
        items[key] = value

    def __delitem__(self, key: K) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def get(self, key, default=None):
        return self._items.get(key, default)

    def clear(self) -> None:
        self._items.clear()

    def __repr__(self) -> str:
        return f"BoundedDict(maxlen={self.maxlen}, {self._items!r})"

    def __reduce__(self):
        return (_rebuild_dict, (self.maxlen, tuple(self._items.items())))


def _rebuild_dict(maxlen: int, items) -> BoundedDict:
    rebuilt: BoundedDict = BoundedDict(maxlen)
    for key, value in items:
        rebuilt[key] = value
    return rebuilt
