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

import hashlib
import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._context import begin_run, end_run
from ._frozendict import FrozenDict
from .dataframe import MetricDataFrame

if TYPE_CHECKING:
    from collections.abc import Callable


# Column names reserved for sample values and orchestrator-provided metadata; a
# run parameter may not reuse them because it would collide in the result frame.
_RESERVED_COLUMNS = ("step", "value", "experiment", "experiment_hash")


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
    run is tagged with the experiment's module-qualified name and a short hash
    of its source, keeping otherwise identical parameter sets distinct.

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
        then tagged with the experiment's module-qualified name and source hash.
        Structured parameter values (dicts, lists, sets) are frozen into
        hashable forms so they can be part of the run key (see :func:`_freeze`).
        The orchestrator publishes that enriched context, invokes
        ``experiment(**params)`` -- which receives the *original* (unfrozen)
        values -- and clears the context afterwards.

        Args:
            experiment: The experiment function, called as
                ``experiment(**params)``.
            **params: The parameter values of this run. Names (including the
                experiment's defaulted arguments) must not include the reserved
                column names ``step``, ``value``, ``experiment``, or
                ``experiment_hash``. Values must be hashable once frozen.

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
        run_params = {**resolved, **self._experiment_params(experiment)}
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
    def _experiment_params(experiment: Callable[..., Any]) -> dict[str, str]:
        """Return stable identifying metadata for ``experiment``.

        The human-readable identity includes the module because qualified names
        alone can collide across modules. The short hash tracks the exact source
        text when inspection is available; otherwise it hashes that identity.

        Args:
            experiment: The callable being run.

        Returns:
            The ``experiment`` name and 12-character ``experiment_hash``.
        """
        module = getattr(experiment, "__module__", type(experiment).__module__)
        qualname = getattr(experiment, "__qualname__", type(experiment).__qualname__)
        experiment_name = f"{module}.{qualname}"
        try:
            hash_input = inspect.getsource(experiment)
        except (OSError, TypeError):
            hash_input = experiment_name
        experiment_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return {
            "experiment": experiment_name,
            "experiment_hash": experiment_hash,
        }

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
