"""Append-only metric recording, and the tidy frame read back from it.

A :class:`Metric` is created inside a running job and finds that job from the
context the orchestrator binds, so a user experiment never has to thread a
handle through its own call stack::

    def my_experiment(n, f, rounds):
        loss = Metric("loss", dtype=float)
        for step in range(rounds):
            loss.push(step, ...)

Rows are flushed as they are pushed, so a crashed job's partial output stays
readable in its staging directory.
See `notes/2026-10-02-orchestrator-v2-a-design.md`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas

from .storage import JobFolder, JobStore, current_job, read_manifest_params

# The columns a metric frame owns. A parameter sharing one of these names
# would be ambiguous in the frame read back, so it is rejected when enqueued.
RESERVED_COLUMNS = ("step", "value", "job_key")


def reserved(params: Iterable[str]) -> list[str]:
    """The parameter names that collide with a metric frame's own columns."""
    return [name for name in params if name in RESERVED_COLUMNS]


class NoActiveJob(RuntimeError):
    """Raised when a metric is created outside a running job.

    A metric has nowhere to write unless the orchestrator is executing a job,
    so this is reported rather than silently buffered in memory and lost.
    """


def coercion(dtype: Any) -> Callable[[Any], Any]:
    """Resolve a declared dtype to a function coercing one recorded value.

    Accepts a Python type such as `float` or `int`, and also a torch dtype,
    which is not callable but reports whether it is floating point.

    Args:
        dtype: The declared metric dtype.

    Returns:
        A callable coercing a pushed value.
    """
    if isinstance(dtype, type):
        return dtype
    is_floating = getattr(dtype, "is_floating_point", None)
    if is_floating is not None:
        return float if is_floating else int
    if callable(dtype):
        return dtype
    return float


def dtype_name(dtype: Any) -> str:
    """Render a declared dtype for the manifest, readably."""
    return getattr(dtype, "__name__", None) or str(dtype)


class Sink:
    """One metric's open file within a running job."""

    _path: Path
    _coerce: Callable[[Any], Any]
    _handle: Any
    _writer: Any
    _seen: set[Any]

    __slots__ = tuple(__annotations__)

    def __init__(self, path: Path, dtype: Any) -> None:
        """Open a metric file and write its header."""
        self._path = path
        self._coerce = coercion(dtype)
        self._handle = path.open("w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(("step", "value"))
        self._handle.flush()
        self._seen = set()

    @property
    def path(self) -> Path:
        """The file this sink writes to."""
        return self._path

    def push(self, step: Any, value: Any, skip_if_exists: bool = False) -> bool:
        """Append one row, coercing the value to the declared dtype.

        Args:
            step: The step the value was measured at.
            value: The measured value.
            skip_if_exists: Leave an already recorded step untouched.

        Returns:
            Whether a row was written.
        """
        if skip_if_exists and step in self._seen:
            return False
        self._seen.add(step)
        self._writer.writerow((step, self._coerce(value)))
        self._handle.flush()
        return True

    def close(self) -> None:
        """Close the underlying file."""
        if not self._handle.closed:
            self._handle.close()


class Metric:
    """A metric recorded by the running job.

    Args:
        name: The metric's name, unrestricted; it is percent-encoded on disk.
        dtype: The type recorded values are coerced to.

    Raises:
        NoActiveJob: If no job is being executed in this context.
    """

    _name: str
    _sink: Sink

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str, dtype: Any = float) -> None:
        """Bind a metric to the running job, reusing its sink if already open."""
        job = current_job()
        if job is None:
            raise NoActiveJob(
                f"metric {name!r} was created outside a job; create it inside the function passed to Orchestrator.run"
            )
        sink = job.sinks.get(name)
        if sink is None:
            sink = Sink(job.metric_path(name, dtype_name(dtype)), dtype)
            job.sinks[name] = sink
        self._name = name
        self._sink = sink

    def __repr__(self) -> str:
        """Render the metric as a constructor call."""
        return f"{type(self).__qualname__}({self._name!r})"

    @property
    def name(self) -> str:
        """The metric's name."""
        return self._name

    def push(self, step: Any, value: Any, skip_if_exists: bool = False) -> bool:
        """Record one value at one step.

        Args:
            step: The step the value was measured at.
            value: The measured value.
            skip_if_exists: Leave an already recorded step untouched.

        Returns:
            Whether a row was written.
        """
        return self._sink.push(step, value, skip_if_exists=skip_if_exists)


def read_metric(folder: JobFolder, name: str) -> pandas.DataFrame | None:
    """Read one metric from one completed job, with its parameters attached.

    Args:
        folder: The job folder to read.
        name: The metric name.

    Returns:
        A frame of `[step, value, *params]`, or None if this job did not record
        that metric.
    """
    path = folder.metric_path(name)
    if not path.is_file():
        return None
    frame = pandas.read_csv(path)
    params = read_manifest_params(folder.manifest())
    clashing = reserved(params)
    if clashing:
        raise ValueError(f"job {folder.key} has parameters shadowing metric columns: {clashing}")
    for parameter, value in params.items():
        frame[parameter] = value
    frame["job_key"] = folder.key
    return frame


def collect(store: JobStore, name: str, keys: Iterable[str] | None = None) -> pandas.DataFrame:
    """Gather one metric across completed jobs in a store.

    Args:
        store: The job store to scan.
        name: The metric name.
        keys: The jobs to read, in the order they should appear. Every
            completed job in the store is read when this is None, which mixes
            code versions: a store accumulates one folder per version of an
            experiment, so two of them can carry the same label and be drawn
            over each other. Passing the keys of one sweep keeps a reading to
            that sweep.

    Returns:
        One tidy frame of `[step, value, *params, job_key]`, the parameters
        being the sweep dimensions, ready for `groupby` or boolean filtering.

    Raises:
        KeyError: If none of the jobs read recorded that metric.
    """
    if keys is None:
        folders: Iterable[JobFolder] = store.done()
    else:
        folders = (folder for folder in map(store.folder_for, keys) if folder.done)
    frames: list[pandas.DataFrame] = [
        frame for frame in (read_metric(folder, name) for folder in folders) if frame is not None
    ]
    if not frames:
        available = store.metric_names()
        raise KeyError(f"no completed job recorded metric {name!r}; available: {available}")
    return pandas.concat(frames, ignore_index=True)


def iter_metrics(store: JobStore) -> Iterator[str]:
    """Iterate over every metric name recorded by a completed job."""
    return iter(store.metric_names())
