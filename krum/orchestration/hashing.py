"""Hashing primitive wrappers and logic for more complex objects."""

from __future__ import annotations

import importlib
import pickle

from collections.abc import Buffer
from hashlib import blake2b as Blake2b
from struct import Struct
from types import FunctionType
from typing import Self

class Location:
    """Serializable object location, by module and qualified name."""

    _module: str
    _path: tuple[str, ...]

    __slots__ = tuple(__annotations__)

    @staticmethod
    def split(qualname: str) -> Iterable[str]:
        MARK = "."
        while len(qualname) > 0:
            pos = qualname.find(MARK)
            if pos < 0:
                pos = len(qualname)
            yield qualname[:pos]
            qualname = qualname[pos + len(MARK):]

    @classmethod
    def of_type(cls, obj: type) -> Self:
        return cls(obj.__module__, obj.__qualname__)

    def __init__(self, module: str, path: str | tuple[str, ...]) -> None:
        if isinstance(path, str):
            path = tuple(self.split(path))
        self._module = module
        self._path = path

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}({self._module!r}, {self._path!r})"

    def __str__(self) -> str:
        path = (".").join(self._path)
        return f"{self._module}.{path}"

    def fetch(self) -> Any:
        object = importlib.import_module(self._module)
        for name in path:
            object = getattr(object, name)
        return object

type Hash = bytes

class Hasher:
    """Higher-level hasher class that can be used with `pickle.dump`."""

    _state: Blake2b

    __slots__ = tuple(__annotations__)

    _LENGHT = Struct("<Q")
    _MODLEN = 1 << (8 * _LENGHT.size)
    _CONTAINERS = {  # type: (special iterator, unordered?, tupled?)
        tuple: (None, False, False),
        list: (None, False, False),
        dict: (dict.items, True, True),
        set: (None, True, False),
        frozenset: (None, True, False) }

    def __init__(self, state: Blake2b | None = None) -> None:
        if state is None:
            state = Blake2b()
        self._state = state

    def copy(self) -> Self:
        return type(self)(self._state.copy())

    def write(self, data: Buffer) -> int:
        self._state.update(data)
        return len(data)

    def _push_length(self, size: int) -> None:
        self._state.update(self._LENGHT.pack(size % self._MODLEN))

    def push(self, obj: Any) -> None:
        # Best-effort substream distinction
        typ = type(obj)
        self._state.update(typ.__module__.encode())
        self._state.update(typ.__qualname__.encode())
        self._state.update(b"\x00")
        # Special case (byte)string
        if isinstance(obj, str):
            self._push_length(len(obj))
            self._state.update(obj.encode())
            return
        if isinstance(obj, Buffer):
            self._push_length(len(obj))
            self._state.update(obj)
            return
        # Special case types and locations
        if isinstance(obj, type):
            self._state.update(obj.__module__.encode())
            self._state.update(b".")
            self._state.update(obj.__qualname__.encode())
            return
        if isinstance(obj, Location):
            self._state.update(obj._module.encode())
            for path in obj._path:
                self._state.update(b".")
                self._state.update(path.encode())
            return
        # Special case containers
        container = self._CONTAINERS.get(typ)
        if container is not None:
            special, unordered, tupled = container
            if special is not None:
                obj = special(obj)
            if unordered:
                obj = sorted(obj)
            if tupled:
                for items in obj:
                    for item in items:
                        self.push(item)
            else:
                for item in obj:
                    self.push(item)
            return
        # Handle function-like objects
        if hasattr(obj, "__code__"):
            raise NotImplementedError
            return
        if hasattr(obj, "__wrapped__"):
            self.push(obj.__wrapped__)
            return
        if hasattr(obj, "__func__"):
            if hasattr(obj, "__self__"):
                self.push(obj.__self__)
            self.push(obj.__func__)
            return
        # Best-effort fallback
        pickle.dump(obj, self)

    def digest(self) -> Hash:
        return self._state.digest()
