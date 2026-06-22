"""Named metric channels the user pushes values into.

A :class:`Metric` is a named channel opened inside the user's experiment
function. Values are recorded one ``(step, value)`` sample at a time::

    loss = Metric("loss", dtype=float)
    for step in range(n_steps):
        simulation.step()
        loss.push(step, simulation.loss())

A :class:`Metric` is a **write handle, not a store**: the samples do not live
on the metric object. Each push is routed to the
:class:`~krum.orchestration.orchestrator.Orchestrator` driving the current run
-- found through the ambient run state (see :mod:`krum.orchestration._context`)
-- which owns and assembles the data. This lets one channel name accumulate
samples across many runs even though a fresh :class:`Metric` is created on every
run.
"""

from __future__ import annotations

from typing import Any

from ._context import active_orchestrator, current_params


class Metric:
    """A named channel of ``(step, value)`` samples written during a run.

    The channel is identified by its ``name``, which is unique per orchestrator
    and case-sensitive (``loss`` and ``Loss`` are distinct). On each push the
    value is recorded together with the current run's parameters, so that
    :meth:`~krum.orchestration.orchestrator.Orchestrator.get` can return every
    sample tagged with the parameters that produced it.
    """

    def __init__(self, name: str, dtype: type = float) -> None:
        """Open (or re-open) the channel ``name`` on the active orchestrator.

        Args:
            name: Channel identifier. Must not contain spaces; case-sensitive.
                Other special characters (``-``, ``_``, ``.``, ``/``) are
                allowed.
            dtype: Declared type of the pushed values. Enforced on every
                :meth:`push`.

        Raises:
            ValueError: If ``name`` contains a space.
            RuntimeError: If created outside of an
                :meth:`~krum.orchestration.orchestrator.Orchestrator.run` call.
            ValueError: If the channel was already declared with a different
                ``dtype``.
        """
        if " " in name:
            raise ValueError(f"Metric name must not contain spaces: {name!r}.")
        self._name = name
        self._dtype = dtype
        # Found now, while a run is active; reused on every push. Registering
        # the channel is delegated to the orchestrator, which owns the data and
        # enforces name/dtype uniqueness across all runs.
        self._orchestrator = active_orchestrator()
        self._orchestrator.register_metric(name, dtype)

    @property
    def name(self) -> str:
        """The channel's name."""
        return self._name

    @property
    def dtype(self) -> type:
        """The channel's declared value type."""
        return self._dtype

    def push(self, step: int, value: Any, skip_if_exists: bool = False) -> None:
        """Record one sample, tagged with the current run's parameters.

        Args:
            step: The step (e.g. training iteration) the value belongs to.
            value: The recorded value. Must be an instance of the declared
                :attr:`dtype`.
            skip_if_exists: If ``True``, the sample is ignored when a value for
                the same run and ``step`` has already been recorded for this
                channel. Lets a partially-completed experiment be re-run
                without producing duplicate rows.

        Raises:
            TypeError: If ``value`` is not an instance of :attr:`dtype`.
            RuntimeError: If called outside of an active run.
        """
        if not isinstance(value, self._dtype):
            raise TypeError(
                f"Metric {self._name!r} expects dtype {self._dtype.__name__}, "
                f"got {type(value).__name__}."
            )
        self._orchestrator.record(
            self._name, current_params(), step, value, skip_if_exists
        )
