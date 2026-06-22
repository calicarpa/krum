"""Per-metric storage and pandas view of collected samples.

A :class:`MetricDataFrame` holds the samples of a single metric. Run parameters
are stored **once per run** -- not repeated on every sample -- and the data is
exposed on demand as a :class:`pandas.DataFrame` indexed by the run parameters,
with ``step`` and ``value`` columns. Calling the instance with keyword filters
returns the matching rows::

    loss_data = orchestrator.get("loss")        # -> MetricDataFrame
    loss_data()                                 # full frame
    curve = loss_data(n=13, aggregator=Krum)    # sliced frame
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class MetricDataFrame:
    """Storage and sliceable pandas view for one metric's samples.

    The samples are kept compactly: each distinct run's parameters are stored
    once, alongside a ``step -> value`` mapping. The
    :class:`pandas.DataFrame` is materialised from that representation only when
    requested, so parameter values are never duplicated in the store.
    """

    def __init__(self, dtype: type = float) -> None:
        """Create an empty store.

        Args:
            dtype: The metric's declared value type (informational here; the
                type is enforced by :class:`~krum.orchestration.metric.Metric`).
        """
        self._dtype = dtype
        # run identity (sorted param items) -> that run's parameters, once.
        self._params: dict[tuple[Any, ...], dict[str, Any]] = {}
        # run identity -> {step: value}.
        self._samples: dict[tuple[Any, ...], dict[int, Any]] = {}

    @property
    def dtype(self) -> type:
        """The metric's declared value type."""
        return self._dtype

    def record(
        self,
        params: dict[str, Any],
        step: int,
        value: Any,
        skip_if_exists: bool = False,
    ) -> None:
        """Store one sample for the run identified by ``params``.

        Args:
            params: The run's parameter values. Stored once per distinct run.
            step: The step the value belongs to.
            value: The recorded value.
            skip_if_exists: If ``True``, ignore the sample when this run already
                has a value for ``step``.
        """
        run_key = tuple(sorted(params.items()))
        if run_key not in self._params:
            self._params[run_key] = dict(params)
            self._samples[run_key] = {}
        steps = self._samples[run_key]
        if skip_if_exists and step in steps:
            return
        steps[step] = value

    def dataframe(self) -> pd.DataFrame:
        """Materialise all samples as a :class:`pandas.DataFrame`.

        Returns:
            A frame with columns ``step`` and ``value``, indexed by the run
            parameters (a :class:`pandas.MultiIndex` when there is more than one
            parameter). Empty when nothing has been recorded.
        """
        param_names = self._param_names()
        rows: list[dict[str, Any]] = []
        index: list[tuple[Any, ...]] = []
        for run_key, steps in self._samples.items():
            params = self._params[run_key]
            for step in sorted(steps):
                rows.append({"step": step, "value": steps[step]})
                index.append(tuple(params[name] for name in param_names))
        if not rows:
            return pd.DataFrame(columns=["step", "value"])
        if not param_names:
            return pd.DataFrame(rows)
        return pd.DataFrame(
            rows, index=pd.MultiIndex.from_tuples(index, names=param_names)
        )

    def _param_names(self) -> list[str]:
        """Return parameter names in the order first seen across runs."""
        names: list[str] = []
        for params in self._params.values():
            for name in params:
                if name not in names:
                    names.append(name)
        return names

    def __call__(self, **filters: Any) -> pd.DataFrame:
        """Return the samples whose run parameters match ``filters``.

        Args:
            **filters: Parameter-name/value pairs all returned rows must match
                (e.g. ``n=13, aggregator=Krum``). With no filters the full
                frame is returned.

        Returns:
            A :class:`pandas.DataFrame` of the matching rows.
        """
        frame = self.dataframe()
        if not filters or frame.empty:
            return frame
        mask = pd.Series(True, index=frame.index)
        for name, value in filters.items():
            mask &= frame.index.get_level_values(name) == value
        return frame[mask]

    def __len__(self) -> int:
        return sum(len(steps) for steps in self._samples.values())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self)} samples, "
            f"dtype={self._dtype.__name__})"
        )
