"""Recording what a job actually called, to catch what a static read cannot see.

A job key covers the code reachable from the experiment *by name*: its own
bytecode, its closure, and the globals it mentions, transitively. What a key
cannot cover is a dependency reached only while running — a module imported
inside a function body, or a helper pulled out of a dict of handlers, or
anything found by `getattr`. Those are invisible to a static read of the
bytecode, so a change to one of them would otherwise go unnoticed, which is
the one failure the orchestrator design rules out.

So a job records which owned functions it actually entered, and the hash of
each one's own code. A later pass fetches them again and re-hashes: anything
renamed, removed or edited makes the stored result stale.

`sys.monitoring` reports the first call to each function and then, told to
`DISABLE`, stops firing for it. The cost is therefore one callback per
distinct function rather than one per call, which is what makes watching a
hundred-round simulation affordable at all.

See `notes/orchestrator-v2-design.md`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from sys import monitoring
from types import CodeType, TracebackType
from typing import Any, Self

from .hashing import Location, Modules, shallow_key

TOOL_NAME = "krum.orchestration"
# 0, 1, 2 and 5 are spoken for by debuggers, coverage, profilers and
# optimizers; 3 and 4 are what is left for everyone else.
TOOL_IDS = (3, 4)
EVENT = monitoring.events.PY_START
# Qualified names holding one of these cannot be fetched back by name, so
# there would be nothing to re-hash later: nested functions, lambdas,
# comprehensions and module-level code.
UNFETCHABLE = "<"


class TracingUnavailable(RuntimeError):
    """Raised when no monitoring tool id is free.

    Python allows only a few tools to watch a process at once, so a debugger,
    profiler or coverage run already in place can leave none for this. Reported
    rather than quietly skipped: a job recorded without its callees would look
    verified when it is not.
    """


class DependencyTracker:
    """Records the owned functions entered while it is active."""

    _owned: Modules
    _codes: set[CodeType]
    _tool: int | None

    __slots__ = tuple(__annotations__)

    def __init__(self, owned: Modules | Iterable[str]) -> None:
        """Watch for calls into a set of owned modules."""
        self._owned = owned if isinstance(owned, Modules) else Modules(owned)
        self._codes = set()
        self._tool = None

    def __repr__(self) -> str:
        """Render the tracker with how much it has seen."""
        return f"{type(self).__qualname__}({self._owned!r}, seen={len(self._codes)})"

    @property
    def active(self) -> bool:
        """Whether monitoring is currently installed."""
        return self._tool is not None

    def _record(self, code: CodeType, offset: int) -> Any:  # noqa: ARG002
        """Note a function on its first call, then stop watching it."""
        self._codes.add(code)
        return monitoring.DISABLE

    def __enter__(self) -> Self:
        """Install monitoring, claiming whichever tool id is free."""
        if self._tool is not None:
            raise RuntimeError("tracker is already active")
        for candidate in TOOL_IDS:
            try:
                monitoring.use_tool_id(candidate, TOOL_NAME)
            except ValueError:
                continue
            self._tool = candidate
            break
        else:
            holders = {identifier: monitoring.get_tool(identifier) for identifier in TOOL_IDS}
            raise TracingUnavailable(f"no monitoring tool id is free, {holders} are in use")
        monitoring.register_callback(self._tool, EVENT, self._record)
        monitoring.set_events(self._tool, EVENT)
        # A `DISABLE` stays in force until events are restarted, and it is
        # recorded against the code object rather than against this tracker.
        # Without this, only the first traced job of a process would see
        # anything: every later one would find its callees already silenced.
        monitoring.restart_events()
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Remove monitoring and release the tool id."""
        tool = self._tool
        if tool is None:
            return
        self._tool = None
        try:
            monitoring.set_events(tool, 0)
            monitoring.register_callback(tool, EVENT, None)
        finally:
            monitoring.free_tool_id(tool)

    def _modules(self) -> dict[str, str]:
        """Map each loaded module's file to its dotted name."""
        found = {}
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                found[origin] = name
        return found

    def called(self) -> dict[str, str]:
        """The owned functions that were entered, by location and code hash.

        A location is hashed through the object fetched back by name, not
        through the code object that was seen running. The two can differ, a
        decorator standing between a name and the code it runs, and it is the
        fetched object that a later pass will re-hash: hashing it on both sides
        is what keeps the comparison meaningful.

        Returns:
            Each callee's encoded location mapped to the hex hash of its code.
        """
        modules = self._modules()
        found: dict[str, str] = {}
        for code in self._codes:
            module = modules.get(code.co_filename)
            if module is None or not self._owned.owns_module(module):
                continue
            qualname = code.co_qualname
            if UNFETCHABLE in qualname:
                continue
            location = Location(module, qualname)
            try:
                fetched = location.fetch()
            except Exception:
                continue
            found[location.encode()] = shallow_key(fetched).hex()
        return dict(sorted(found.items()))


def verify_called(recorded: Mapping[str, str]) -> list[str]:
    """Explain why any function a job called no longer matches.

    Args:
        recorded: Encoded locations mapped to the code hashes a job recorded.

    Returns:
        One reason per callee that has changed or gone, empty when they all
        still match.
    """
    reasons = []
    for encoded, digest in sorted(recorded.items()):
        location = Location.decode(encoded)
        try:
            fetched = location.fetch()
        except Exception:
            reasons.append(f"{encoded} is no longer there")
            continue
        if shallow_key(fetched).hex() != digest:
            reasons.append(f"{encoded} changed")
    return reasons
