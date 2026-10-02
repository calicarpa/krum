"""Hashing primitive wrappers and logic for more complex objects.

The entry point is :func:`static_key`, which derives a run's identity from the
user callable's code and its bound parameters without executing anything. See
`notes/orchestrator-v2-design.md` for the surrounding design.

Two properties are deliberate:

- File and line provenance is excluded, so adding a comment above a function,
  or moving it within its file, does not invalidate a job.
- Recursion stops at the boundary of the *owned* modules (see :class:`Modules`).
  Owned code is folded in by content; everything else stops at its
  :class:`Location`. Dependency *versions* are not hashed here: they belong to
  the environment fingerprint, keyed on ``uv.lock``.

Bytecode is not stable across interpreter versions, so keys change on a Python
upgrade. The interpreter version is part of the environment fingerprint, so
this is visible rather than silent.
"""

from __future__ import annotations

import importlib
import inspect
import pickle
from collections.abc import Buffer, Iterable, Iterator, Mapping
from contextvars import ContextVar
from hashlib import blake2b as Blake2b
from struct import Struct
from types import (
    BuiltinFunctionType,
    CodeType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodDescriptorType,
    MethodWrapperType,
    ModuleType,
    WrapperDescriptorType,
)
from typing import Any, Self


class HashError(Exception):
    """Raised when an object cannot be folded into a hash reproducibly.

    Failing loudly is intentional: silently folding in a placeholder would hide
    a changed dependency, and a false negative (a stale result kept) is the one
    outcome the orchestrator design rules out.
    """


class Location:
    """Serializable object location, by module and qualified name."""

    _module: str
    _path: tuple[str, ...]

    __slots__ = tuple(__annotations__)

    @staticmethod
    def split(qualname: str) -> Iterable[str]:
        """Split a dotted qualified name into its components."""
        MARK = "."
        while len(qualname) > 0:
            pos = qualname.find(MARK)
            if pos < 0:
                pos = len(qualname)
            yield qualname[:pos]
            qualname = qualname[pos + len(MARK):]

    @classmethod
    def of_type(cls, obj: type) -> Self:
        """Locate a class by its module and qualified name."""
        return cls(obj.__module__, obj.__qualname__)

    @classmethod
    def of_callable(cls, obj: Any) -> Self:
        """Locate a callable, falling back on placeholders when it is anonymous."""
        module = getattr(obj, "__module__", None)
        name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?")
        return cls(module or "?", name)

    def __init__(self, module: str, path: str | tuple[str, ...]) -> None:
        """Build a location from a module name and a qualified name or path."""
        if isinstance(path, str):
            path = tuple(self.split(path))
        self._module = module
        self._path = path

    def __repr__(self) -> str:
        """Render the location as a constructor call."""
        return f"{type(self).__qualname__}({self._module!r}, {self._path!r})"

    def __str__(self) -> str:
        """Render the location as a dotted path."""
        path = (".").join(self._path)
        return f"{self._module}.{path}"

    @classmethod
    def decode(cls, encoded: str) -> Self:
        """Rebuild a location from its string form."""
        module, _, path = encoded.partition(":")
        return cls(module, path)

    def encode(self) -> str:
        """Render the location so that it can be parsed back.

        The module and the qualified name are separated by a colon, a dotted
        form being ambiguous: `a.b.C.d` could be module `a.b` or module `a`.
        """
        path = (".").join(self._path)
        return f"{self._module}:{path}"

    @property
    def module(self) -> str:
        """The dotted name of the module holding the object."""
        return self._module

    @property
    def qualname(self) -> str:
        """The qualified name of the object within its module."""
        return (".").join(self._path)

    def fetch(self) -> Any:
        """Import and return the located object."""
        object = importlib.import_module(self._module)
        for name in self._path:
            object = getattr(object, name)
        return object


class Modules:
    """Ownership test over module names, by dotted prefix.

    Owned modules are folded in by content: bytecode, closures, referenced
    globals, class bodies. Everything else stops at its :class:`Location`,
    which is what keeps a hash of a user experiment from walking into pytorch.

    Exclusions override the prefixes, which is what lets a broad prefix like
    `krum` be owned while a subpackage within it is not. The orchestration
    package itself is always excluded: it *runs* a job rather than defining
    what the job computes, so folding it in would both change every key
    whenever the harness is edited and drag in runtime state, such as the
    context variable naming the current job, that has no reproducible hash.
    """

    _prefixes: tuple[str, ...]
    _exclude: tuple[str, ...]

    __slots__ = tuple(__annotations__)

    @staticmethod
    def _matches(name: str, prefixes: Iterable[str]) -> bool:
        return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)

    def __init__(self, prefixes: Iterable[str], exclude: Iterable[str] = ()) -> None:
        """Build an ownership test from dotted module prefixes and exclusions."""
        self._prefixes = tuple(sorted(set(prefixes)))
        self._exclude = tuple(sorted({*exclude, __name__.rpartition(".")[0]}))

    def __repr__(self) -> str:
        """Render the ownership test as a constructor call."""
        return f"{type(self).__qualname__}({self._prefixes!r}, {self._exclude!r})"

    @property
    def prefixes(self) -> tuple[str, ...]:
        """The owned module prefixes, sorted."""
        return self._prefixes

    @property
    def exclude(self) -> tuple[str, ...]:
        """The excluded module prefixes, sorted, which override the owned ones."""
        return self._exclude

    def owns_module(self, name: str | None) -> bool:
        """Whether a dotted module name is owned, exclusions taking precedence."""
        if not name:
            return False
        return self._matches(name, self._prefixes) and not self._matches(name, self._exclude)

    def __contains__(self, obj: Any) -> bool:
        """Whether an object, or a module itself, belongs to an owned module."""
        if isinstance(obj, ModuleType):
            return self.owns_module(obj.__name__)
        return self.owns_module(getattr(obj, "__module__", None))


type Hash = bytes


class Hasher:
    """Higher-level hasher class that can be used with `pickle.dump`."""

    _state: Blake2b
    _owned: Modules
    _seen: dict[int, int]

    __slots__ = tuple(__annotations__)

    # Digest size in bytes. 16 is ample to name a job folder, and keeps the
    # folder name and the `job_key` column readable.
    _DIGEST_SIZE = 16
    _LENGHT = Struct("<Q")
    _MODLEN = 1 << (8 * _LENGHT.size)
    _CONTAINERS = {  # type: (special iterator, unordered?, tupled?)
        tuple: (None, False, False),
        list: (None, False, False),
        dict: (dict.items, True, True),
        set: (None, True, False),
        frozenset: (None, True, False) }
    # Members that carry no behaviour of their own; see `_push_class`. The slot
    # and weakref descriptors are machinery, `_abc_impl` is an identity-based
    # ABCMeta cache already covered by the method bodies, and `__firstlineno__`
    # (3.13+) is the line provenance this module excludes everywhere else.
    _CLASS_SKIP = frozenset({"__dict__", "__weakref__", "_abc_impl", "__firstlineno__"})
    # Objects implemented in C, folded in by name only
    _OPAQUE = (
        BuiltinFunctionType,
        GetSetDescriptorType,
        MemberDescriptorType,
        MethodDescriptorType,
        MethodWrapperType,
        WrapperDescriptorType,
    )
    # Substream markers
    _MARK_REVISIT = b"\x01"
    _MARK_VISIT = b"\x02"
    _MARK_EMPTY = b"\x03"

    def __init__(self, owned: Modules | Iterable[str] | None = None, *, state: Blake2b | None = None) -> None:
        """Build a hasher.

        Args:
            owned: Modules to fold in by content, as prefixes or a
                :class:`Modules`. Defaults to `__main__` alone.
            state: An existing hash state to continue from.
        """
        if owned is None:
            owned = Modules(("__main__",))
        elif not isinstance(owned, Modules):
            owned = Modules(owned)
        if state is None:
            state = Blake2b(digest_size=self._DIGEST_SIZE)
        self._state = state
        self._owned = owned
        self._seen = {}

    @property
    def owned(self) -> Modules:
        """The ownership test bounding how far recursion goes."""
        return self._owned

    def copy(self) -> Self:
        """Return an independent hasher continuing from this one's state."""
        clone = type(self)(self._owned, state=self._state.copy())
        clone._seen = dict(self._seen)
        return clone

    def write(self, data: Buffer) -> int:
        """Fold raw bytes in, so that `pickle.dump` can write to this hasher."""
        self._state.update(data)
        # A buffer need not be sized; a memoryview of it always is
        return memoryview(data).nbytes

    def _push_length(self, size: int) -> None:
        self._state.update(self._LENGHT.pack(size % self._MODLEN))

    def _enter(self, obj: Any) -> bool:
        """Mark an object as visited, returning False if it was folded in already.

        A revisit folds in the first visit's order instead of recursing, so the
        hash still reflects the shape of a cyclic object graph.
        """
        oid = id(obj)
        order = self._seen.get(oid)
        if order is not None:
            self._state.update(self._MARK_REVISIT)
            self._push_length(order)
            return False
        self._seen[oid] = len(self._seen)
        self._state.update(self._MARK_VISIT)
        return True

    def push(self, obj: Any) -> None:
        """Fold an object into the running hash.

        Raises:
            HashError: If the object, or something it reaches inside the owned
                modules, cannot be hashed reproducibly.
        """
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
        # Special case locations
        if isinstance(obj, Location):
            self._state.update(obj._module.encode())
            for path in obj._path:
                self._state.update(b".")
                self._state.update(path.encode())
            return
        # Special case code, classes and modules, which carry their own identity
        if isinstance(obj, CodeType):
            self._push_code(obj)
            return
        if isinstance(obj, type):
            self._push_class(obj)
            return
        if isinstance(obj, ModuleType):
            self._push_module(obj)
            return
        # Special case containers
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
            self._push_function(obj)
            return
        if hasattr(obj, "__wrapped__"):
            self.push(obj.__wrapped__)
            return
        if hasattr(obj, "__func__"):
            if hasattr(obj, "__self__"):
                self.push(obj.__self__)
            self.push(obj.__func__)
            return
        if isinstance(obj, property):
            for accessor in (obj.fget, obj.fset, obj.fdel):
                self.push(accessor)
            return
        # Objects implemented in C cannot be introspected; their name is all we have
        if isinstance(obj, self._OPAQUE):
            self.push(Location.of_callable(obj))
            return
        # A context variable names a slot for runtime state; the name is its
        # static identity, and what it happens to hold is not static at all
        if isinstance(obj, ContextVar):
            self.push(obj.name)
            return
        # An instance of an owned class: fold the class body in, then the instance state
        if typ in self._owned:
            self.push(typ)
        # Best-effort fallback
        try:
            pickle.dump(obj, self)
        except Exception as error:
            raise HashError(f"cannot reproducibly hash {typ.__qualname__} object: {obj!r}") from error

    @classmethod
    def _code_names(cls, code: CodeType) -> Iterator[str]:
        """Yield the names a code object references, nested code objects included.

        Nested code matters because a lambda or an inner function resolves its
        globals against the *enclosing* function's `__globals__`.
        """
        yield from code.co_names
        for const in code.co_consts:
            if isinstance(const, CodeType):
                yield from cls._code_names(const)

    def _push_code(self, code: CodeType) -> None:
        """Fold in a code object's structure, excluding its file and line provenance.

        `co_filename`, `co_firstlineno` and the line table are left out on
        purpose, so that reformatting or relocating a function is not a change.
        """
        for count in (
            code.co_argcount,
            code.co_posonlyargcount,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_flags,
        ):
            self._push_length(count)
        self._push_length(len(code.co_code))
        self._state.update(code.co_code)
        self.push(code.co_name)
        for names in (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars):
            self._push_length(len(names))
            for name in names:
                self.push(name)
        # Constants include nested code objects, docstrings and default literals
        self._push_length(len(code.co_consts))
        for const in code.co_consts:
            self.push(const)

    def _push_function(self, obj: Any) -> None:
        """Fold in a function: location, bytecode, defaults, closure and referenced globals."""
        if not self._enter(obj):
            return
        location = Location.of_callable(obj)
        self.push(location)
        # A function outside the owned set stops at its location
        if not self._owned.owns_module(location.module):
            return
        code = obj.__code__
        self._push_code(code)
        self.push(obj.__defaults__ or ())
        self.push(obj.__kwdefaults__ or {})
        # Closure cells, which hold the values the function captured
        cells = obj.__closure__ or ()
        self._push_length(len(cells))
        for cell in cells:
            try:
                contents = cell.cell_contents
            except ValueError:
                # Cell not filled yet, e.g. a function still being defined
                self._state.update(self._MARK_EMPTY)
                continue
            self.push(contents)
        # The globals the function names. This is what makes a transitive change
        # to a helper, a class or a constant reachable without running anything.
        globals = getattr(obj, "__globals__", None) or {}
        names = sorted(set(self._code_names(code)) & globals.keys())
        self._push_length(len(names))
        for name in names:
            self.push(name)
            self.push(globals[name])

    def _push_class(self, obj: type) -> None:
        """Fold in a class: its location, and for an owned class its bases and body."""
        if not self._enter(obj):
            return
        self.push(Location.of_type(obj))
        if obj not in self._owned:
            return
        bases = obj.__mro__[1:]
        self._push_length(len(bases))
        for base in bases:
            self.push(base)
        members = sorted((name, value) for name, value in vars(obj).items() if name not in self._CLASS_SKIP)
        self._push_length(len(members))
        for name, value in members:
            self.push(name)
            self.push(value)

    def _push_module(self, obj: ModuleType) -> None:
        """Fold in a module by name, and for an owned module its public members.

        Recursing into an owned module covers the `import mymod; mymod.helper()`
        shape, where the dependency is reached by attribute rather than by a
        name in the caller's globals.
        """
        if not self._enter(obj):
            return
        self.push(obj.__name__)
        if obj not in self._owned:
            return
        members = sorted((name, value) for name, value in vars(obj).items() if not name.startswith("__"))
        self._push_length(len(members))
        for name, value in members:
            self.push(name)
            self.push(value)

    def digest(self) -> Hash:
        """The hash of everything folded in so far."""
        return self._state.digest()


def code_of(obj: Any) -> CodeType | None:
    """Find the code object behind a callable, unwrapping what hides it.

    Methods, class and static methods, and functions wrapped by
    `functools.wraps` all stand between a name and the code it runs.

    Args:
        obj: The callable to look through.

    Returns:
        Its code object, or None if it has none, as for a builtin.
    """
    for _ in range(16):
        code = getattr(obj, "__code__", None)
        if code is not None:
            return code if isinstance(code, CodeType) else None
        following = getattr(obj, "__func__", None) or getattr(obj, "__wrapped__", None)
        if following is None:
            return None
        obj = following
    return None


def shallow_key(obj: Any) -> Hash:
    """Hash a callable's own code, and nothing it refers to.

    Where :func:`static_key` folds in everything an experiment reaches, this
    isolates one function, so that a change can be attributed to it rather
    than to the whole closure around it. Line provenance is excluded here too,
    so reformatting is not a change.

    Args:
        obj: The callable to hash.

    Returns:
        The hash of its code, or of the object itself when it has none.
    """
    hasher = Hasher(())
    code = code_of(obj)
    hasher.push(obj if code is None else code)
    return hasher.digest()


def bind_params(callable: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Bind parameters to a callable's signature and apply its defaults.

    The job key and the recorded manifest are both derived from this, so that
    what a result is traced back to is exactly what its identity was computed
    from.

    Args:
        callable: The user-defined function a run executes.
        params: The hyper parameter values, by name.

    Returns:
        The bound parameters, in signature order, defaults included.

    Raises:
        TypeError: If the parameters do not fit the signature.
    """
    bound = inspect.signature(callable).bind(**params)
    bound.apply_defaults()
    return dict(bound.arguments)


def static_key(callable: Any, params: Mapping[str, Any], owned: Modules | Iterable[str] | None = None) -> Hash:
    """Derive a run's static identity from its callable and its parameters.

    The parameters are bound to the signature and defaults applied, so that
    parameter *names* participate: renaming a hyper parameter yields a distinct
    key, as does renaming the callable, changing its body, or changing anything
    it reaches inside the owned modules.

    Nothing is executed. Raises `TypeError` if the parameters do not fit the
    signature, and :class:`HashError` on a dependency that cannot be hashed
    reproducibly.

    Args:
        callable: The user-defined function a run would execute.
        params: The hyper parameter values, by name.
        owned: Modules to fold in by content, as prefixes or a :class:`Modules`.

    Returns:
        The key identifying this run, used to name its output folder.
    """
    hasher = Hasher(owned)
    hasher.push(callable)
    hasher.push(bind_params(callable, params))
    return hasher.digest()
