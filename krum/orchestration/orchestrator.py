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
from typing import TYPE_CHECKING, Any

from ._context import begin_run, end_run
from .dataframe import MetricDataFrame

if TYPE_CHECKING:
    from collections.abc import Callable


# Column names reserved for the sample itself; a run parameter may not reuse
# them, otherwise it would collide with a metric column in the result frame.
_RESERVED_COLUMNS = ("step", "value")


class Orchestrator:
    """Runs experiments and owns the metric data they produce.

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
        defaulted argument was passed explicitly. The orchestrator publishes the
        resolved parameters as the current run context, invokes
        ``experiment(**params)``, and clears the context afterwards. Any
        :class:`~krum.orchestration.metric.Metric` pushed during the call is
        tagged with those parameters.

        Args:
            experiment: The experiment function, called as
                ``experiment(**params)``.
            **params: The parameter values of this run. Names (including the
                experiment's defaulted arguments) must not include the reserved
                column names ``step`` or ``value``.

        Raises:
            ValueError: If a parameter name collides with a reserved column.
            RuntimeError: If ``experiment`` raises. The error names the failing
                run's parameters and chains the original exception (fail-fast:
                the sweep stops). The run context is cleared either way.
        """
        resolved = self._resolve_params(experiment, params)
        conflicts = sorted(set(resolved) & set(_RESERVED_COLUMNS))
        if conflicts:
            raise ValueError(
                f"Parameter name(s) {conflicts} are reserved for metric columns."
            )
        begin_run(self, resolved)
        try:
            # Future work: dispatch to a worker process (one per run) instead
            # of calling inline; the orchestrator will also handle the PRNG seed.
            experiment(**params)
        except Exception as error:
            # Fail-fast: re-raise so the sweep stops, but name the failing run
            # so it can be diagnosed. The original exception is chained.
            raise RuntimeError(f"Run failed for params {resolved}.") from error
        finally:
            end_run()

    @staticmethod
    def _resolve_params(
        experiment: Callable[..., Any], params: dict[str, Any]
    ) -> dict[str, Any]:
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
