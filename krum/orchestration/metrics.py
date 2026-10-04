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
See `notes/adr-2026-10-02-orchestrator-v2-a-design.md`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .storage import JobFolder, JobStore, PathLike, current_job, read_manifest_params

if TYPE_CHECKING:  # read by type checkers; never imported at runtime
    import pandas

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


class MetricTable:
    """One metric's rows, column by column.

    Deliberately not a dataframe. It holds plain Python lists and leaves the
    analysis to whichever library the reader prefers, so that recording a
    metric costs no dependency at all::

        table = orchestrator.get("loss")
        table.columns          # ('step', 'value', 'n', 'f', 'job_key')
        table["value"]         # [1.2, 0.7, ...]
        len(table)             # the number of rows
        for row in table:      # one dict per row
            ...

    :meth:`to_pandas` and :meth:`to_csv` are provided, and are written against
    :meth:`to_dict` alone, so a converter you write yourself has exactly the
    access the built-in ones do::

        import polars
        polars.DataFrame(table.to_dict())

        import pyarrow
        pyarrow.table(table.to_dict())

        import numpy
        {name: numpy.asarray(values) for name, values in table.to_dict().items()}

    Args:
        columns: One list of values per column, all of the same length.

    Raises:
        ValueError: If the columns are not all the same length.
    """

    _columns: dict[str, list[Any]]

    __slots__ = tuple(__annotations__)

    def __init__(self, columns: Mapping[str, list[Any]]) -> None:
        """Hold one list per column, checking that they line up."""
        lengths = {name: len(values) for name, values in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"columns have differing lengths: {lengths}")
        self._columns = {name: list(values) for name, values in columns.items()}

    def __repr__(self) -> str:
        """Render the table's shape."""
        return f"{type(self).__qualname__}({len(self)} rows, columns={self.columns})"

    def __len__(self) -> int:
        """The number of rows."""
        return len(next(iter(self._columns.values()))) if self._columns else 0

    def __getitem__(self, column: str) -> list[Any]:
        """One column's values."""
        return list(self._columns[column])

    def __contains__(self, column: str) -> bool:
        """Whether a column is present."""
        return column in self._columns

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the rows, one dict each, as :meth:`rows` does."""
        return self.rows()

    @property
    def columns(self) -> tuple[str, ...]:
        """The column names, in order."""
        return tuple(self._columns)

    def to_dict(self) -> dict[str, list[Any]]:
        """The columns as plain lists.

        This is the accessor every converter is built on, the shipped ones
        included.
        """
        return {name: list(values) for name, values in self._columns.items()}

    def rows(self) -> Iterator[dict[str, Any]]:
        """Iterate over the rows, one dict each."""
        names = tuple(self._columns)
        for values in zip(*self._columns.values(), strict=True):
            yield dict(zip(names, values, strict=True))

    def to_pandas(self) -> pandas.DataFrame:
        """Convert to a `pandas.DataFrame`.

        Returns:
            A frame of the same columns, in the same order.

        Raises:
            ImportError: If pandas is not installed. It is not a requirement of
                this library; the error says what to do instead.
        """
        try:
            import pandas
        except ImportError as error:
            raise ImportError(
                "to_pandas needs pandas, which krum does not require: install it, "
                "or hand to_dict() to the dataframe library you prefer"
            ) from error
        return pandas.DataFrame(self.to_dict())

    def to_csv(self, path: PathLike) -> Path:
        """Write the table as one CSV file.

        Args:
            path: Where to write it.

        Returns:
            The path written.
        """
        destination = Path(path)
        with destination.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(tuple(row.values()) for row in self.rows())
        return destination


def parse_value(recorded: str) -> Callable[[str], Any]:
    """Resolve a recorded dtype name to a function parsing one CSV field.

    The dtype a metric was declared with is kept in the manifest, so a value
    is parsed back as what it was written as rather than guessed at. A torch
    dtype is recorded by name, and reads back as the Python type it stands
    for.

    Args:
        recorded: The dtype name from the manifest, such as `float` or
            `torch.float32`.

    Returns:
        A callable parsing one field.
    """
    if recorded in {"float", "float64", "float32", "float16"}:
        return float
    if recorded in {"int", "int64", "int32", "int16", "int8"}:
        return int
    if recorded == "bool":
        return lambda field: field == "True"
    if recorded == "str":
        return str
    if recorded.startswith("torch."):
        return int if "int" in recorded or "bool" in recorded else float
    return infer_value


def infer_value(field: str) -> Any:
    """Parse one field whose type was not recorded, trying int then float."""
    for parse in (int, float):
        try:
            return parse(field)
        except ValueError:
            continue
    return field


def read_metric(folder: JobFolder, name: str) -> MetricTable | None:
    """Read one metric from one completed job, with its parameters attached.

    Args:
        folder: The job folder to read.
        name: The metric name.

    Returns:
        A table of `[step, value, *params, job_key]`, or None if this job did
        not record that metric.

    Raises:
        ValueError: If the job's parameters would shadow the table's own
            columns.
    """
    path = folder.metric_path(name)
    if not path.is_file():
        return None
    manifest = folder.manifest()
    params = read_manifest_params(manifest)
    clashing = reserved(params)
    if clashing:
        raise ValueError(f"job {folder.key} has parameters shadowing metric columns: {clashing}")
    parse = parse_value(manifest.get("metrics", {}).get(name, {}).get("dtype", ""))
    steps: list[Any] = []
    values: list[Any] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # the header, written by `Sink`
        for row in reader:
            if not row:
                continue
            steps.append(infer_value(row[0]))
            values.append(parse(row[1]))
    columns: dict[str, list[Any]] = {"step": steps, "value": values}
    for parameter, value in params.items():
        columns[parameter] = [value] * len(steps)
    columns["job_key"] = [folder.key] * len(steps)
    return MetricTable(columns)


def concat(tables: Iterable[MetricTable]) -> MetricTable:
    """Stack tables, taking the union of their columns.

    Jobs in one sweep need not share a parameter set, so a column absent from
    a table is filled with None for its rows rather than dropped. Nothing is
    coerced: an integer parameter stays an integer where it was recorded, which
    a dataframe would not promise once a column holds a gap.

    Args:
        tables: The tables to stack, in order.

    Returns:
        One table holding every row.
    """
    collected = list(tables)
    seen: dict[str, None] = {}
    for table in collected:
        seen.update(dict.fromkeys(table.columns))
    # Keep the documented shape even when the tables disagree on parameters:
    # the metric's own columns first, then the parameters, then the job key.
    leading = [name for name in ("step", "value") if name in seen]
    trailing = [name for name in ("job_key",) if name in seen]
    middle = [name for name in seen if name not in {*leading, *trailing}]
    names = [*leading, *middle, *trailing]
    columns: dict[str, list[Any]] = {name: [] for name in names}
    for table in collected:
        height = len(table)
        for name in names:
            columns[name].extend(table[name] if name in table else [None] * height)
    return MetricTable(columns)


def collect(store: JobStore, name: str, keys: Iterable[str] | None = None) -> MetricTable:
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
        One tidy table of `[step, value, *params, job_key]`, the parameters
        being the sweep dimensions. Hand it to a dataframe library through
        :meth:`MetricTable.to_pandas` or :meth:`MetricTable.to_dict` to group
        or filter it.

    Raises:
        KeyError: If none of the jobs read recorded that metric.
    """
    if keys is None:
        folders: Iterable[JobFolder] = store.done()
    else:
        folders = (folder for folder in map(store.folder_for, keys) if folder.done)
    tables = [table for table in (read_metric(folder, name) for folder in folders) if table is not None]
    if not tables:
        available = store.metric_names()
        raise KeyError(f"no completed job recorded metric {name!r}; available: {available}")
    return concat(tables)


def iter_metrics(store: JobStore) -> Iterator[str]:
    """Iterate over every metric name recorded by a completed job."""
    return iter(store.metric_names())
