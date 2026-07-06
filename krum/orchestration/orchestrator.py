"""The :class:`Orchestrator`: drives runs and owns the collected metric data.

The orchestrator is the entry point of an experiment campaign. The user creates
one, then calls :meth:`Orchestrator.run` once per parameter point -- writing the
parameter sweep as ordinary Python loops::

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
    loss = orch.get("loss")   # -> MetricDataFrame

The orchestrator registers channels and owns their data, but the data itself
lives in a :class:`~krum.orchestration.dataframe.MetricDataFrame` per channel;
the orchestrator does not build pandas frames itself. Data is held in memory and
never persisted; create a new orchestrator to start fresh. Execution is
**fail-fast**: if an experiment raises, the exception propagates and the
campaign stops.

This is the synchronous draft. The near-term plan (multi-process, one process
per run, plus PRNG seed handling) will change *how* :meth:`run` dispatches work,
but not the :class:`~krum.orchestration.metric.Metric` / :meth:`get` contract.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ._frozendict import FrozenDict
from .dataframe import MetricDataFrame

# Column names reserved for sample values and orchestrator-provided metadata; a
# run parameter may not reuse them because it would collide in the result frame.
_RESERVED_COLUMNS = ("step", "value", "experiment")


def _freeze(value: Any) -> Any:
    """Return a hashable, immutable form of a run-parameter ``value``.

    Run parameters form the immutable run key, so each value must be hashable.
    Mappings become :class:`~krum.orchestration._frozendict.FrozenDict`, lists
    and tuples become tuples, and sets become frozensets -- recursively -- so a
    structured parameter such as ``attack_kwargs={...}`` can be part of the key.
    Already-hashable values (numbers, strings, classes, ...) are returned
    unchanged. A genuinely unhashable leaf (e.g. a tensor) is returned as-is and
    is rejected later by the hashability check in :meth:`Orchestrator.run`.

    The freezing lives here rather than in ``FrozenDict`` on purpose: on Python
    3.15 ``FrozenDict`` is the built-in ``frozendict``, which does not freeze its
    values, so doing it here keeps behaviour identical across versions.
    """
    if isinstance(value, Mapping):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _is_hashable(value: Any) -> bool:
    """Return whether ``value`` can be hashed."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


class Orchestrator:
    """Runs experiments and owns the metric data they produce.

    An orchestrator can drive multiple experiments and parameter sweeps. Every
    run is tagged with the experiment function itself (under the ``experiment``
    key), so otherwise-identical parameter sets from different experiments stay
    distinct.

    A channel is identified by its name, unique within the orchestrator. Each
    channel's samples are stored in a
    :class:`~krum.orchestration.dataframe.MetricDataFrame`, returned by
    :meth:`get`.
    """

    def __init__(self, name: str) -> None:
        """Create an empty campaign.

        Args:
            name: A unique identifier for this orchestrator. Used to tell
                orchestrators apart when debugging; not used as a metric prefix
                or namespace in this version.
        """
        self.name = name
        # Channel name -> its storage.
        self._frames: dict[str, MetricDataFrame] = {}

    # -- public API -------------------------------------------------------

    def run(self, experiment: Callable[..., Any], **params: Any) -> None:
        """Run ``experiment`` once at a single parameter point.

        The parameters are first enriched with the experiment's default
        arguments (see :meth:`_resolve_params`), so two runs of the same
        experiment are tagged with the same parameter set whether or not a
        defaulted argument was passed explicitly. The resolved parameters are
        then tagged with the experiment function itself (under the ``experiment``
        key). Structured parameter values (dicts, lists, sets) are frozen into
        hashable forms so they can be part of the run key (see :func:`_freeze`).
        The orchestrator publishes that enriched context, invokes
        ``experiment(**params)`` -- which receives the *original* (unfrozen)
        values -- and clears the context afterwards.

        Args:
            experiment: The experiment function, called as
                ``experiment(**params)``.
            **params: The parameter values of this run. Names (including the
                experiment's defaulted arguments) must not include the reserved
                column names ``step``, ``value``, or ``experiment``. Values must
                be hashable once frozen.

        Raises:
            ValueError: If a parameter name collides with a reserved column, or
                if a parameter value is not hashable even after freezing.
            RuntimeError: If ``experiment`` raises. The error names the failing
                run's parameters and chains the original exception (fail-fast:
                the sweep stops). The run context is cleared either way.
        """
        resolved = self._resolve_params(experiment, params)
        conflicts = sorted(set(resolved) & set(_RESERVED_COLUMNS))
        if conflicts:
            raise ValueError(f"Parameter name(s) {conflicts} are reserved for metric columns.")
        run_params = {**resolved, "experiment": experiment}
        run_params = {name: _freeze(value) for name, value in run_params.items()}
        unhashable = sorted(name for name, value in run_params.items() if not _is_hashable(value))
        if unhashable:
            raise ValueError(
                f"Run parameter(s) {unhashable} have unhashable values and cannot "
                f"form the run key. Pass hashable values (numbers, strings, "
                f"classes, tuples, or nested dicts/lists of those)."
            )
        begin_run(self, run_params)
        try:
            # Future work: dispatch to a worker process (one per run) instead
            # of calling inline; the orchestrator will also handle the PRNG seed.
            experiment(**params)
        except Exception as error:
            # Fail-fast: re-raise so the sweep stops, but name the failing run
            # so it can be diagnosed. The original exception is chained.
            raise RuntimeError(f"Run failed for params {run_params}.") from error
        finally:
            end_run()

    @staticmethod
    def _resolve_params(experiment: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
        """Enrich ``params`` with ``experiment``'s default arguments.

        Binding the call against the experiment's signature and applying its
        defaults means a defaulted argument is recorded the same way whether the
        caller passed it explicitly or relied on the default -- so runs that
        differ only in that respect share one parameter set (and one identity).

        Args:
            experiment: The experiment function whose signature is inspected.
            params: The parameter values passed to :meth:`run`.

        Returns:
            ``params`` plus any defaulted arguments. Returned unchanged if the
            signature cannot be inspected or bound (e.g. a built-in, or a
            mismatched call -- in which case the call itself surfaces the error).
        """
        try:
            bound = inspect.signature(experiment).bind(**params)
        except (TypeError, ValueError):
            return params
        bound.apply_defaults()
        return dict(bound.arguments)

    def get(self, name: str) -> MetricDataFrame:
        """Return the storage for channel ``name``.

        Args:
            name: The channel name.

        Returns:
            The channel's :class:`~krum.orchestration.dataframe.MetricDataFrame`,
            which exposes the samples as a pandas frame and can be sliced by
            parameter values.

        Raises:
            KeyError: If no channel called ``name`` was ever declared.
        """
        if name not in self._frames:
            raise KeyError(f"No metric named {name!r} was declared.")
        return self._frames[name]

    def metrics(self) -> list[str]:
        """Return the names of all declared channels."""
        return list(self._frames)

    # -- internals used by Metric -----------------------------------------

    def register_metric(self, name: str, dtype: type) -> None:
        """Declare a channel, enforcing per-orchestrator name/dtype uniqueness.

        Idempotent: re-declaring an existing name with the same ``dtype`` is a
        no-op (this happens once per run of a sweep, since the metric is created
        inside the experiment function).

        Args:
            name: Channel name.
            dtype: Declared value type.

        Raises:
            ValueError: If ``name`` was already declared with a different
                ``dtype``.
        """
        existing = self._frames.get(name)
        if existing is not None:
            if existing.dtype != dtype:
                raise ValueError(
                    f"Metric {name!r} already declared with dtype "
                    f"{existing.dtype.__name__}; cannot redeclare with "
                    f"{dtype.__name__}."
                )
            return
        self._frames[name] = MetricDataFrame(dtype)

    def record(
        self,
        name: str,
        params: dict[str, Any],
        step: int,
        value: Any,
        skip_if_exists: bool,
    ) -> None:
        """Store one sample of channel ``name`` for the given run parameters.

        Args:
            name: Channel name (already registered).
            params: The current run's parameter values.
            step: The step the value belongs to.
            value: The recorded value.
            skip_if_exists: If ``True`` and a sample for the same run and
                ``step`` already exists, do nothing.
        """
        self._frames[name].record(params, step, value, skip_if_exists)


# -- Ambient run state, used by Metric to reach the orchestrator ------------
#
# The state of the run currently executing is held in **plain module-level
# globals**. There is no stack and no thread/async isolation: exactly one run
# is active at a time -- the near-term execution model is multi-process, one
# process per run, with no multithreading -- so a simple global is sufficient
# and a :class:`contextvars.ContextVar` would be needless machinery.
#
# Two pieces of state are tracked while a run executes: the active
# orchestrator, so a :class:`~krum.orchestration.metric.Metric` created inside
# the experiment can reach the orchestrator that owns the data; and the
# current parameters, used to tag every pushed sample. Both are set at the
# start of a run by :func:`begin_run` and cleared at its end by
# :func:`end_run`.

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
        The active :class:`Orchestrator`.

    Raises:
        RuntimeError: If no run is active, which usually means a
            :class:`~krum.orchestration.metric.Metric` was created outside of
            :meth:`Orchestrator.run`.
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
            pushed outside of :meth:`Orchestrator.run`.
    """
    if _current_params is None:
        raise RuntimeError(
            "No active run. Metric.push(...) may only be called inside a function executed by Orchestrator.run(...)."
        )
    return _current_params
