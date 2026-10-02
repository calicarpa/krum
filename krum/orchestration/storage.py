"""On-disk job folder protocol: identity-addressed outputs, with provenance.

Layout under the orchestrator's root::

    <root>/<job_key>/
        manifest.json   parameters, callable location, provenance, timings
        deps.json       the environment fingerprint
        metrics/*.csv   append-only (step, value), one file per metric
        DONE | FAILED   completion marker; FAILED carries the traceback

A job's state is read from its marker rather than from a mutable field, and a
job is built in a staging directory that is renamed into place only once its
marker is written. A job killed mid-flight therefore leaves no marker at the
final path and is simply "not done" on the next pass: there is no folder that
is incomplete yet marked complete, and no lock to reason about.

See `notes/orchestrator-v2-design.md`.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import blake2b as Blake2b
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote, unquote

from .hashing import Hash

# Marker and member names inside a job folder
DONE = "DONE"
FAILED = "FAILED"
MANIFEST = "manifest.json"
DEPS = "deps.json"
METRICS = "metrics"
# Staging area, a sibling of the job folders so that renames stay on one filesystem
STAGING = ".staging"

# A plain alias rather than a `type` statement, which needs Python 3.12
PathLike = str | os.PathLike[str]


def encode_value(value: Any) -> Any:
    """Encode a parameter value as JSON, keeping enough to trace it back.

    Scalars pass through. A class or function becomes a tagged object carrying
    both its full location, for traceability, and its short name, which is what
    reads well as a column value.

    Args:
        value: The parameter value to encode.

    Returns:
        A JSON-serializable encoding of the value.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, type):
        return {"$kind": "class", "location": f"{value.__module__}.{value.__qualname__}", "name": value.__qualname__}
    if callable(value) and hasattr(value, "__qualname__"):
        location = f"{getattr(value, '__module__', '?')}.{value.__qualname__}"
        return {"$kind": "callable", "location": location, "name": value.__qualname__}
    if isinstance(value, list | tuple):
        return [encode_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): encode_value(item) for key, item in value.items()}
    return {"$kind": "opaque", "location": type(value).__qualname__, "name": repr(value)}


def display_value(encoded: Any) -> Any:
    """Render an encoded parameter as a single cell value.

    Args:
        encoded: A value as returned by :func:`encode_value`.

    Returns:
        The scalar itself, a tagged value's short name, or a compact string.
    """
    if isinstance(encoded, dict):
        if "$kind" in encoded:
            return encoded["name"]
        return json.dumps(encoded, sort_keys=True)
    if isinstance(encoded, list):
        return json.dumps(encoded)
    return encoded


def encode_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Encode a parameter mapping, preserving its order."""
    return {name: encode_value(value) for name, value in params.items()}


def find_lock(start: PathLike | None = None) -> Path | None:
    """Find the nearest `uv.lock` at or above a directory.

    Args:
        start: Where to start looking; the current directory by default.

    Returns:
        The path to `uv.lock`, or None if there is none above `start`.
    """
    current = Path(start) if start is not None else Path.cwd()
    current = current if current.is_dir() else current.parent
    for directory in (current, *current.parents):
        candidate = directory / "uv.lock"
        if candidate.is_file():
            return candidate
    return None


def hash_file(path: Path) -> str:
    """Hash a file's contents, as a hex digest."""
    state = Blake2b()
    with path.open("rb") as handle:
        buffer = memoryview(bytearray(65536))
        while True:
            read = handle.readinto(buffer)
            if read == 0:
                break
            state.update(buffer[:read])
    return state.hexdigest()


def environment_fingerprint(lock: PathLike | None = None, start: PathLike | None = None) -> dict[str, Any]:
    """Describe the environment a job ran in.

    The lock file hash stands in for the resolved dependency versions, which is
    exact and cheap next to sniffing each distribution. This lives in the
    fingerprint rather than in the job key, so that a dependency bump marks
    stored results stale without orphaning them under a new key.

    Args:
        lock: The lock file to hash; discovered from `start` when omitted.
        start: Where to look for a lock file; the current directory by default.

    Returns:
        A JSON-serializable description of the environment. A lock file that
        cannot be found, or that does not exist, is recorded as None rather
        than raising: an environment without one is still worth describing.
    """
    path = Path(lock) if lock is not None else find_lock(start)
    if path is not None and not path.is_file():
        path = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "debug": __debug__,
        "platform": platform.platform(),
        "uv_lock": {"path": str(path), "hash": hash_file(path)} if path is not None else None,
    }


def git_provenance(start: PathLike | None = None) -> dict[str, Any] | None:
    """Describe the working tree a job ran from, so an output can be traced back.

    A dirty tree is recorded rather than rejected: the orchestrator warns, and
    leaves it to the user to decide whether the result is publishable.

    Args:
        start: A directory inside the repository; the current one by default.

    Returns:
        The commit and whether the tree was dirty, or None outside a repository.
    """
    cwd = Path(start) if start is not None else Path.cwd()

    def git(*arguments: str) -> str | None:
        try:
            done = subprocess.run(
                ("git", *arguments), cwd=cwd, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--porcelain")
    return {
        "commit": commit.strip(),
        "dirty": bool(status.strip()) if status is not None else None,
        "branch": (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
    }


def metric_filename(name: str) -> str:
    """Encode a metric name as a file name, reversibly.

    Metric names are unrestricted, so they are percent-encoded rather than
    sanitized, keeping the mapping unambiguous in both directions.
    """
    return f"{quote(name, safe='')}.csv"


def metric_name(filename: str) -> str:
    """Recover a metric name from its file name."""
    return unquote(filename[: -len(".csv")] if filename.endswith(".csv") else filename)


class JobFolder:
    """One job's directory on disk, and the questions one can ask of it."""

    _path: Path

    __slots__ = tuple(__annotations__)

    def __init__(self, path: PathLike) -> None:
        """Wrap a job directory, which need not exist yet."""
        self._path = Path(path)

    def __repr__(self) -> str:
        """Render the folder as a constructor call."""
        return f"{type(self).__qualname__}({str(self._path)!r})"

    @property
    def path(self) -> Path:
        """The job directory."""
        return self._path

    @property
    def key(self) -> str:
        """The job key naming this directory."""
        return self._path.name

    @property
    def status(self) -> str:
        """One of `done`, `failed` or `absent`, read from the markers."""
        if (self._path / DONE).is_file():
            return "done"
        if (self._path / FAILED).is_file():
            return "failed"
        return "absent"

    @property
    def done(self) -> bool:
        """Whether this job completed successfully."""
        return self.status == "done"

    def manifest(self) -> dict[str, Any]:
        """Read the job's manifest."""
        return json.loads((self._path / MANIFEST).read_text())

    def deps(self) -> dict[str, Any]:
        """Read the job's fingerprint."""
        return json.loads((self._path / DEPS).read_text())

    def traceback(self) -> str | None:
        """The recorded traceback of a failed job, if any."""
        marker = self._path / FAILED
        return marker.read_text() if marker.is_file() else None

    def metric_path(self, name: str) -> Path:
        """The file holding one metric's rows, whether or not it exists."""
        return self._path / METRICS / metric_filename(name)

    def metric_names(self) -> list[str]:
        """The metrics this job recorded, sorted."""
        directory = self._path / METRICS
        if not directory.is_dir():
            return []
        return sorted(metric_name(entry.name) for entry in directory.iterdir() if entry.name.endswith(".csv"))


class JobWriter:
    """Builds one job in a staging directory, then promotes it atomically.

    Metrics are flushed as they are pushed, so a crashed job's partial output
    stays readable in the staging directory, without ever being promoted to the
    job's final path where the read path would pick it up.
    """

    _store: JobStore
    _key: str
    _path: Path
    _manifest: dict[str, Any]
    _metrics: dict[str, dict[str, Any]]
    _sinks: dict[str, Any]
    _started: float

    __slots__ = tuple(__annotations__)

    def __init__(self, store: JobStore, key: str, manifest: Mapping[str, Any]) -> None:
        """Create the staging directory for a job and record its manifest."""
        self._store = store
        self._key = key
        self._path = store.staging_path(key)
        if self._path.exists():
            shutil.rmtree(self._path)
        (self._path / METRICS).mkdir(parents=True)
        self._manifest = dict(manifest)
        self._metrics = {}
        self._sinks = {}
        self._started = perf_counter()

    @property
    def path(self) -> Path:
        """The staging directory this job is being built in."""
        return self._path

    @property
    def key(self) -> str:
        """The job key."""
        return self._key

    @property
    def sinks(self) -> dict[str, Any]:
        """The open metric writers, by metric name.

        Held here rather than on the metric objects so that constructing the
        same metric twice inside one job keeps appending to one file.
        """
        return self._sinks

    def metric_path(self, name: str, dtype: str) -> Path:
        """Register a metric and return the file it writes to."""
        self._metrics[name] = {"dtype": dtype, "file": metric_filename(name)}
        return self._path / METRICS / metric_filename(name)

    def finish(self, status: str, error: str | None = None) -> JobFolder:
        """Write the manifest, the marker and the fingerprint, then promote.

        Args:
            status: Either `done` or `failed`.
            error: The traceback to record in the `FAILED` marker.

        Returns:
            The promoted job folder, at its final path.
        """
        for sink in self._sinks.values():
            sink.close()
        manifest = dict(self._manifest)
        manifest["status"] = status
        manifest["metrics"] = dict(sorted(self._metrics.items()))
        manifest["timings"] = {
            **manifest.get("timings", {}),
            "finished": datetime.now(timezone.utc).isoformat(),
            "seconds": round(perf_counter() - self._started, 6),
        }
        (self._path / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
        (self._path / DEPS).write_text(json.dumps(manifest.get("environment", {}), indent=2) + "\n")
        marker = DONE if status == "done" else FAILED
        (self._path / marker).write_text(error or "")
        return self._store.promote(self)

    def discard(self) -> None:
        """Remove the staging directory without promoting it."""
        shutil.rmtree(self._path, ignore_errors=True)


class JobStore:
    """The orchestrator's root directory, holding one folder per job."""

    _root: Path

    __slots__ = tuple(__annotations__)

    def __init__(self, root: PathLike) -> None:
        """Wrap a root directory, creating it if needed."""
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Render the store as a constructor call."""
        return f"{type(self).__qualname__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The root directory."""
        return self._root

    @staticmethod
    def name_for(key: Hash | str) -> str:
        """The folder name for a job key."""
        return key if isinstance(key, str) else key.hex()

    def folder_for(self, key: Hash | str) -> JobFolder:
        """The folder a job key maps to, whether or not it exists."""
        return JobFolder(self._root / self.name_for(key))

    def staging_path(self, key: Hash | str) -> Path:
        """The staging directory a job is built in, beside the job folders."""
        return self._root / STAGING / f"{self.name_for(key)}.{os.getpid()}"

    def writer(self, key: Hash | str, manifest: Mapping[str, Any]) -> JobWriter:
        """Start building a job."""
        return JobWriter(self, self.name_for(key), manifest)

    def promote(self, writer: JobWriter) -> JobFolder:
        """Move a staged job to its final path, replacing any earlier attempt.

        The replace is a rename, so the final path never holds a partially
        written job. Removing an earlier attempt first is safe: we only get
        here having decided to re-run it, and a crash in between leaves no
        marker, which the next pass reads as "not done".
        """
        folder = self.folder_for(writer.key)
        if folder.path.exists():
            shutil.rmtree(folder.path)
        folder.path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(writer.path, folder.path)
        return folder

    def __iter__(self) -> Iterator[JobFolder]:
        """Iterate over every job folder, in key order."""
        for entry in sorted(self._root.iterdir()):
            if entry.is_dir() and entry.name != STAGING:
                yield JobFolder(entry)

    def done(self) -> Iterator[JobFolder]:
        """Iterate over the job folders that completed successfully."""
        return (folder for folder in self if folder.done)

    def metric_names(self) -> list[str]:
        """Every metric name recorded by a completed job, sorted."""
        names: set[str] = set()
        for folder in self.done():
            names.update(folder.metric_names())
        return sorted(names)

    def prune_staging(self) -> int:
        """Remove leftover staging directories from crashed jobs.

        Returns:
            The number of directories removed.
        """
        staging = self._root / STAGING
        if not staging.is_dir():
            return 0
        removed = 0
        for entry in staging.iterdir():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed


_CURRENT: ContextVar[JobWriter | None] = ContextVar("krum_current_job", default=None)


def current_job() -> JobWriter | None:
    """The job being executed in this context, if any."""
    return _CURRENT.get()


@contextmanager
def bind_job(writer: JobWriter) -> Iterator[JobWriter]:
    """Make a job current for the duration of a block."""
    token = _CURRENT.set(writer)
    try:
        yield writer
    finally:
        _CURRENT.reset(token)


def build_manifest(
    key: Hash | str,
    callable: Any,
    params: Mapping[str, Any],
    *,
    owned: Iterable[str] = (),
    lock: PathLike | None = None,
    start: PathLike | None = None,
) -> dict[str, Any]:
    """Assemble the manifest recorded alongside a job's output.

    Args:
        key: The job key.
        callable: The user-defined function the job executes.
        params: The bound parameters, in signature order.
        owned: The module prefixes the key was computed over.
        lock: The lock file to fingerprint.
        start: A directory inside the repository to read provenance from, and
            to discover the lock file from.

    Returns:
        The manifest, ready to be written as JSON.
    """
    return {
        "job_key": JobStore.name_for(key),
        "callable": {
            "module": getattr(callable, "__module__", None),
            "qualname": getattr(callable, "__qualname__", None),
        },
        "params": encode_params(params),
        "owned": sorted(owned),
        "git": git_provenance(start),
        "environment": environment_fingerprint(lock, start),
        "timings": {"started": datetime.now(timezone.utc).isoformat()},
    }


def read_manifest_params(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest's parameters, rendered one cell value each."""
    return {name: display_value(value) for name, value in manifest.get("params", {}).items()}
