"""Run experiments over parameter ranges and collect their metrics.

The user writes an experiment as a single run, sweeps it with ordinary Python
loops, and reads the results back per metric.

Metric values are collected in memory and not persisted. Execution is
synchronous and fail-fast in this version; the near-term plan is multi-process,
one process per run.

Example::

    from krum.orchestration import Metric, Orchestrator

    def my_experiment(n, f, aggregator, attack, n_steps):
        simulation = KrumSimulation(n=n, f=f, aggregator=aggregator, attack=attack)
        loss = Metric("loss", dtype=float)
        for step in range(n_steps):
            simulation.step()
            loss.push(step, simulation.loss())

    orch = Orchestrator("byzantine_study")
    for n in [10, 20]:
        for f in [2, 3]:
            for aggregator in [Average, Krum, Bulyan]:
                for attack in [ALIEAttack, SignFlipAttack, None]:
                    orch.run(
                        my_experiment,
                        n=n, f=f, aggregator=aggregator, attack=attack,
                        n_steps=100,
                    )

    loss = orch.get("loss")               # MetricDataFrame
    krum_alie = loss.filter(aggregator=Krum, attack=ALIEAttack)  # narrowed MetricDataFrame
    frame = krum_alie.to_pandas()         # pandas.DataFrame for plotting/analysis
"""

from __future__ import annotations

import gc
import importlib
import inspect
import sys

from collections import deque as Deque
from collections.abc import Callable
from hashlib import blake2b as Blake2b
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from inspect import Parameter
from pathlib import Path, PurePath
from types import CodeType, FrameType, ModuleType, TracebackType
from typing import Any, Self

class Modules:
    """Collection of modules for ownership testing purpose."""

    _modules: set[ModuleType]

    __slots__ = tuple(__annotations__)

    @classmethod
    def build(cls, modules_or_names: Iterable[str | ModuleType]) -> Self:
        modules = set()
        # Import names and add modules
        for module_or_name in modules_or_names:
            if isinstance(module_or_name, str):
                module = importlib.import_module(module_or_name)
            else:
                module = module_or_name
            modules.add(module)
        # Initialize members
        self._modules = modules

    def __init__(self, modules: set[ModuleType]) -> None:
        self._modules = modules

    def __contains__(self, obj: Any) -> bool:
        return inspect.getmodule(obj) in self._modules

class Context:
    """Playground context."""

    _external: set[ModuleType]
    _internal: set[CodeType | ObjectRef]

    __slots__ = tuple(__annotations__)

    def trace(self, frame: FrameType, event: str, arg: Any | None) -> Callable | None:
        # Quickly process non-call events
        match event:
            case "call":
                pass
            case "line":
                return self.trace
            case "return":
                return
            case "opcode":
                return self.trace
            case "exception":
                return self.trace
            case _:
                raise RuntimeError(f"unknown event {event!r}")
        # NOTE: Debug
        return_value = None
        import code
        code.interact(local=locals())
        return return_value

class Dependencies:
    """(Conservative) list of dependencies for a collection of objects."""

    _modules: dict[str, int]
    _hash: int | None

    __slots__ = tuple(__annotations__)

    # Fixed hash size (in bytes)
    _HASH_SIZE: int = 16
    # Runtime hash of the current interpreter (computed by `__preinit__`)
    _HASH_BASE: int

    @classmethod
    def _bytes_to_int(cls, data: bytes) -> int:
        value = 0
        for byte in data[:cls._HASH_SIZE]:
            value = value * 2**8 + byte
        return value

    @classmethod
    def _hash_spec(cls, spec: ModuleSpec) -> int:
        # Recover actual location and open file
        origin = Path(spec.origin)
        while True:
            try:
                fd = origin.open("rb")
                break
            except NotADirectoryError:
                # Handle ZIP container (e.g. /path/to/container.zip/package/submodule.py)
                origin = origin.parent
        # Hash module name and "content" together
        with fd:
            b2b = Blake2b()
            b2b.update(spec.name.encode())
            b2b.update(b"\x00")
            buf = memoryview(bytearray(65536))
            while True:
                read = fd.readinto(buf)
                if read == 0:
                    del buf
                    break
                b2b.update(buf[:read])
        # Digest and forward
        return cls._bytes_to_int(b2b.digest())

    @classmethod
    def __preinit__(cls) -> None:
        # Hash about current interpreter
        b2b = Blake2b()
        b2b.update(sys.version.encode())
        b2b.update(b"\xfe" if __debug__ else b"\xff")
        cls._HASH_BASE = cls._bytes_to_int(b2b.digest())

    @classmethod
    def derive(cls, closure: Callable) -> Self:
        # TODO: Do something more relevant than only hashing code; only resort to hashing code
        #       for native functions (and just-in-time imports, for lack of a better solution).
        #       Process non-native closures/generators by discovering relevant referent objects,
        #       and only hashing bytecode (which should be stable for each interpreter version).
        #       Hash other objects based on their pickled stream (optionally with fixed version).
        #       (Acknowledge this is fundamentally an impossible problem, c.f. `exec(random())`.)
        #       The end-user will not like to see everything run again after each micro-change.
        import code
        code.interact(local=locals())
        # Extract function signature
        signature = inspect.signature(closure)
        signature.parameters
        # Process whole referent tree
        todo = Deque()
        todo.append(closure)
        seen = set()
        origins = set()
        while True:
            # Pull next object to process
            try:
                obj = todo.popleft()
            except IndexError:
                break
            # Ensure objects are seen at most once
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            # Print sub-referents
            refs = gc.get_referents(obj)
            print(f"Referents of 0x{oid:016x} (type {type(obj).__qualname__}):")
            cnt = len(refs)
            spc = 0
            while cnt > 0:
                spc += 1
                cnt //= 10
            for idx, ref in enumerate(refs):
                oid = id(ref)
                addum = ", seen" if oid in seen else ""
                print(f"- refs[{idx:{spc}}] = 0x{oid:016x} (type {type(ref).__qualname__}{addum})")
            # Interact with sub-referents
            import code
            code.interact(local=locals())
            # Prepare to process sub-referents
            todo.extend(refs)
        # TODO:
        raise NotImplementedError

    def __init__(self, modules: dict[str, int]) -> None:
        # Compute hash
        hash = self._HASH_BASE
        for module in modules.values():
            hash ^= module
        # Initialize members
        self._modules = modules
        self._hash = hash

    @property
    def hash(self) -> int:
        return self._hash

    def modules(self) -> Iterator[str]:
        return iter(self._modules)

    def push(self, module: str, spec: ModuleSpec) -> None:
        hash = self._hash_spec(spec)
        prev = self._modules.get(module)
        if prev is None:
            self._modules[module] = hash
            self._hash ^= hash
        elif hash != prev:
            raise RuntimeError(f"trying to overwrite hash of {module!r}")

# Finalize class initialization
Dependencies.__preinit__()

class InterceptFinder(MetaPathFinder):
    """Meta path finder intercepting and integrating new imports."""

    _target: Dependencies
    _finders: list[MetaPathFinder] | None

    __slots__ = tuple(__annotations__)

    def __init__(self, target: Dependencies) -> None:
        self._target = target
        self._finders = None

    def __enter__(self) -> None:
        if self._finders is not None:
            raise RuntimeError("unsupported reentrancy")
        self._finders = sys.meta_path
        sys.meta_path = [self]

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        sys.meta_path = self._finders
        self._finders = None

    def find_spec(self, fullname: str, path: str | None, target: ModuleType | None = None) -> ModuleSpec | None:
        for finder in self._finders:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                self._target.push(fullname, spec)
                return spec

type RunCallable = Callable[..., None]

class PendingRun:
    """Information about an enqueued run."""

    _callable: RunCallable
    _params: dict[str, Any]
    _key: Hash | None

    __slots__ = tuple(__annotations__)

    def __init__(self, callable: RunCallable, params: dict[str, Any], key: Hash | None = None) -> None:
        self._callable = callable
        self._params = params
        self._key = key

    @property
    def callable(self) -> RunCallable:
        return self._callable

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    @property
    def key(self) -> Hash:
        if self._key is not None:
            return self._key
        raise NotImplementedError

class Orchestrator:
    """Top-most orchestrator instance managing runs and persisting metrics."""

    _root: Path
    _queue: Deque[PendingRun]

    __slots__ = tuple(__annotations__)

    def __init__(self, root: PathLike) -> None:
        if not isinstance(root, Path):
            root = Path(root)
        self._root = root

    def run(self, callable: Callable[..., Any], **params: Any) -> Any:
        self._queue.append(PendingRun(callable, params))
        raise NotImplementedError

    def get(self, metric: str) -> MetricReader:
        # NOTE: May be blocking
        raise NotImplementedError
