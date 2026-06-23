"""Immutable, hashable mapping -- a backport of :pep:`814`'s ``frozendict``.

:pep:`814` adds a built-in ``frozendict`` type in Python 3.15. Until this
project's minimum supported version reaches 3.15 this module provides a
compatible :class:`FrozenDict`; on 3.15 and later it transparently aliases the
built-in instead, so call sites never change.

:class:`FrozenDict` behaves like :class:`dict` but is immutable and hashable, so
it can serve as a dictionary key or set member. Following ``dict`` semantics,
equality and hashing ignore insertion order while iteration preserves it. Keys
must be hashable; values may be unhashable, in which case hashing the mapping
raises :class:`TypeError` (lazily, only when it is actually hashed) -- matching
the built-in.
"""

from __future__ import annotations

import builtins
import copy
from collections.abc import Iterable, Iterator, Mapping
from reprlib import recursive_repr
from types import MappingProxyType
from typing import Any, TypeVar

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class FrozenDict(Mapping[_Key, _Value]):
    """An immutable, hashable mapping; see the module docstring."""

    __slots__ = ("__data", "__hash")

    def __init__(
        self,
        mapping: Mapping[_Key, _Value] | Iterable[tuple[_Key, _Value]] = (),
        /,
        **kwargs: Any,
    ) -> None:
        """Create a frozen mapping from a mapping, iterable, and/or keywords.

        Args:
            mapping: Initial mapping or iterable of key/value pairs, in order.
            **kwargs: Additional items, applied after ``mapping``.
        """
        data = dict(mapping)
        data.update(kwargs)
        object.__setattr__(self, "_FrozenDict__data", MappingProxyType(data))
        object.__setattr__(self, "_FrozenDict__hash", None)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__!s} is immutable")

    def __getitem__(self, key: _Key) -> _Value:
        return self.__data[key]

    def __iter__(self) -> Iterator[_Key]:
        return iter(self.__data)  # insertion order

    def __len__(self) -> int:
        return len(self.__data)

    def __hash__(self) -> int:
        # Computed on demand and cached. Like the built-in, values may be
        # unhashable; the error then surfaces here, when the mapping is
        # hashed, rather than at construction.
        cached = self.__hash
        if cached is None:
            cached = hash(frozenset(self.__data.items()))
            object.__setattr__(self, "_FrozenDict__hash", cached)
        return cached

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.__data == other  # order-independent
        return NotImplemented

    def __or__(self, other: object) -> FrozenDict[Any, Any]:
        if not isinstance(other, Mapping):
            return NotImplemented
        return type(self)(self.__data | dict(other))

    def __ror__(self, other: object) -> FrozenDict[Any, Any]:
        if not isinstance(other, Mapping):
            return NotImplemented
        return type(self)(dict(other) | self.__data)

    def __ior__(self, other: object) -> FrozenDict[Any, Any]:
        return self | other

    def copy(self) -> FrozenDict[_Key, _Value]:
        """Return this instance, since it is immutable."""
        return self

    def __copy__(self) -> FrozenDict[_Key, _Value]:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict[_Key, _Value]:
        return type(self)(copy.deepcopy(dict(self.__data), memo))

    def __reduce__(self) -> tuple[type[object], tuple[dict[_Key, _Value]]]:
        return type(self), (dict(self.__data),)

    @recursive_repr()
    def __repr__(self) -> str:
        items = ", ".join(f"{key!r}: {value!r}" for key, value in self.__data.items())
        return f"{type(self).__name__}({{{items}}})"


if hasattr(builtins, "frozendict"):  # Python 3.15+ (PEP 814)
    FrozenDict = builtins.frozendict  # type: ignore[misc,assignment]
