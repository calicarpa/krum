"""Internal run state shared between :class:`Orchestrator` and :class:`Metric`.

The state of the run currently executing is held in **plain module-level
globals**. There is no stack and no thread/async isolation: exactly one run is
active at a time -- the near-term execution model is multi-process, one process
per run, with no multithreading -- so a simple global is sufficient and a
:class:`contextvars.ContextVar` would be needless machinery.

Two pieces of state are tracked while a run executes:

* the **current context** -- the parameter values of the running experiment,
  used to tag every pushed sample;
* the **active orchestrator** -- so that a :class:`Metric` created inside the
  experiment can reach the orchestrator that owns the data.

Both are set at the start of a run by :func:`begin_run` and cleared at its end
by :func:`end_run`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

_active_orchestrator: Orchestrator | None = None
_current_params: dict[str, Any] | None = None


def begin_run(orchestrator: Orchestrator, params: dict[str, Any]) -> None:
    """Mark the start of a run, publishing its orchestrator and parameters.

    Args:
        orchestrator: The orchestrator driving the run.
        params: The parameter values of the run, as a mapping of name to value.
    """
    global _active_orchestrator, _current_params
    _active_orchestrator = orchestrator
    _current_params = params


def end_run() -> None:
    """Mark the end of a run, clearing the published state.

    No stack is kept: the state simply returns to "no run active".
    """
    global _active_orchestrator, _current_params
    _active_orchestrator = None
    _current_params = None


def active_orchestrator() -> Orchestrator:
    """Return the orchestrator of the run currently executing.

    Returns:
        The active :class:`~krum.orchestration.orchestrator.Orchestrator`.

    Raises:
        RuntimeError: If no run is active, which usually means a
            :class:`Metric` was created outside of
            :meth:`~krum.orchestration.orchestrator.Orchestrator.run`.
    """
    if _active_orchestrator is None:
        raise RuntimeError(
            "No active orchestrator. A Metric may only be created inside a function executed by Orchestrator.run(...)."
        )
    return _active_orchestrator


def current_params() -> dict[str, Any]:
    """Return the parameter values of the run currently executing.

    Returns:
        The current run's parameters, as a mapping of name to value.

    Raises:
        RuntimeError: If no run is active, which usually means a value was
            pushed outside of
            :meth:`~krum.orchestration.orchestrator.Orchestrator.run`.
    """
    if _current_params is None:
        raise RuntimeError(
            "No active run. Metric.push(...) may only be called inside a function executed by Orchestrator.run(...)."
        )
    return _current_params
