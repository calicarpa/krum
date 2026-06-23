"""Per-metric storage and pandas view of collected samples.

A :class:`MetricDataFrame` holds the samples of a single metric. Run parameters
are stored once per run, and :meth:`to_pandas` materialises them as flat columns
alongside ``step`` and ``value``. Calling the instance with keyword filters
returns a narrowed :class:`MetricDataFrame`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._frozendict import FrozenDict

# Sentinel used by filtering to distinguish an absent parameter from a
# parameter explicitly set to ``None``.
_MISSING = object()


class MetricDataFrame:
    """Compact storage and filtering for one metric's samples."""

    def __init__(self, dtype: type = float) -> None:
        """Create an empty store.

        Args:
            dtype: The metric's declared value type (informational here; the
                type is enforced by :class:`~krum.orchestration.metric.Metric`).
        """
        self._dtype = dtype
        # Immutable run parameters identify a run and are stored once; each
        # value maps to that run's step-to-metric-value samples.
        self._samples: dict[FrozenDict[str, Any], dict[int, Any]] = {}

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
            params: The run's parameter values. Stored once as an immutable key.
            step: The step the value belongs to.
            value: The recorded value.
            skip_if_exists: If ``True``, ignore the sample when this run already
                has a value for ``step``.
        """
        run_key = FrozenDict(params)
        steps = self._samples.setdefault(run_key, {})
        if skip_if_exists and step in steps:
            return
        steps[step] = value

    def to_pandas(self) -> pd.DataFrame:
        """Materialise all samples as a flat :class:`pandas.DataFrame`.

        Parameter columns are the ordered union of parameter names across runs.
        A run missing a parameter receives :data:`pandas.NA` in that column. The final
        columns are always ``step`` and ``value``.

        Returns:
            A fresh frame with parameter, ``step``, and ``value`` columns.
        """
        param_names = list(dict.fromkeys(name for params in self._samples for name in params))
        columns = [*param_names, "step", "value"]
        rows: list[dict[str, Any]] = []
        for params, steps in self._samples.items():
            for step in sorted(steps):
                rows.append({
                    # ``pd.NA`` marks a parameter absent from this run, unlike
                    # an explicit parameter whose value is Python ``None``.
                    **{name: params.get(name, pd.NA) for name in param_names},
                    "step": step,
                    "value": steps[step],
                })
        frame = pd.DataFrame(rows, columns=columns)
        for name in param_names:
            # Object dtype preserves ``pd.NA`` while retaining dtype inference
            # for the step and value columns.
            frame[name] = pd.Series([row[name] for row in rows], dtype=object)
        return frame

    def __call__(self, **filters: Any) -> MetricDataFrame:
        """Return a narrowed store whose run parameters match ``filters``.

        Args:
            **filters: Parameter-name/value pairs that every selected run must
                match. Unknown names produce an empty store.

        Returns:
            An independent :class:`MetricDataFrame` containing matching runs.
        """
        narrowed = type(self)(self._dtype)
        for params, steps in self._samples.items():
            matches = all(params.get(name, _MISSING) == value for name, value in filters.items())
            if matches:
                narrowed._samples[params] = dict(steps)
        return narrowed

    def __len__(self) -> int:
        """Return the number of samples across all stored runs."""
        return sum(len(steps) for steps in self._samples.values())

    def __repr__(self) -> str:
        """Return a concise summary of this metric store."""
        return f"{type(self).__name__}({len(self)} samples, dtype={self._dtype.__name__})"
