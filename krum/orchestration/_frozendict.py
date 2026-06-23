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
from collections.abc import Iterator, Mapping
from typing import Any

if hasattr(builtins, "frozendict"):  # Python 3.15+ (PEP 814)
    FrozenDict = builtins.frozendict
else:

    class FrozenDict(Mapping):
        """An immutable, hashable mapping; see the module docstring."""

        def __init__(
            self, mapping: Mapping[str, Any] | None = None, /, **kwargs: Any
        ) -> None:
            """Create a frozen mapping from a mapping and/or keyword items.

            Args:
                mapping: Initial items, in order.
                **kwargs: Additional items, applied after ``mapping``.
            """
            data: dict[str, Any] = dict(mapping or {})
            data.update(kwargs)
            self._data = data
            self._hash: int | None = None

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._data)  # insertion order

        def __len__(self) -> int:
            return len(self._data)

        def __hash__(self) -> int:
            # Computed on demand and cached. Like the built-in, values may be
            # unhashable; the error then surfaces here, when the mapping is
            # hashed, rather than at construction.
            if self._hash is None:
                self._hash = hash(frozenset(self._data.items()))
            return self._hash

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Mapping):
                return dict(self._data) == dict(other)  # order-independent
            return NotImplemented

        def __repr__(self) -> str:
            items = ", ".join(
                f"{key!r}: {value!r}" for key, value in self._data.items()
            )
            return f"{type(self).__name__}({{{items}}})"
