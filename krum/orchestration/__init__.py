"""Run experiments over parameter ranges, persisting their metrics.

The user writes an experiment as a single run, sweeps it with ordinary Python
loops, and reads the results back per metric as a tidy `pandas.DataFrame`.

Each run is a *job*, identified by its parameters and by the code it executes
(see :mod:`krum.orchestration.hashing`), and owning a folder under the
orchestrator's root (see :mod:`krum.orchestration.storage`). A job is
re-executed only when it has no recorded result, when its last attempt failed,
when the environment it was recorded in no longer matches, or when one of the
functions it called has since changed; otherwise its stored output stands.
`Orchestrator.plan` reports what a sweep would do, and why, without running
any of it.

Execution is synchronous and fail-fast: the first failing job stops the sweep,
and the remaining jobs stay queued. One process per job, and re-running a job
whose dependencies changed, are the next two steps in
`notes/2026-10-02-orchestrator-v2-a-design.md`.

Example::

    from krum.orchestration import Metric, Orchestrator
    from krum.primitives.aggregators.average import Average
    from krum.primitives.aggregators.krum import Krum
    from krum.primitives.attacks.alie import ALIEAttack
    from krum.primitives.attacks.sign_flip import SignFlipAttack

    def my_experiment(n, f, aggregator, attack, seed):
        simulation = ...  # e.g. a krum.simulations class
        loss = Metric("loss", dtype=float)
        for step in range(100):
            simulation.step()
            if step % 10 == 0:
                test_loss, _accuracy = simulation.evaluate()
                loss.push(step, test_loss)

    with Orchestrator("byzantine_study") as orch:
        for n, f in [(10, 2), (20, 3)]:
            for aggregator in [Average, Krum]:
                for attack in [ALIEAttack, SignFlipAttack]:
                    orch.run(my_experiment, n=n, f=f, aggregator=aggregator, attack=attack, seed=42)

    loss = orch.get("loss")  # columns: step, value, n, f, aggregator, attack, seed, job_key
    krum_alie = loss[(loss["aggregator"] == "Krum") & (loss["attack"] == "ALIEAttack")]
    mean_loss = loss.groupby(["aggregator", "attack", "step"])["value"].mean()
"""

from __future__ import annotations

from collections import deque as Deque
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any, Self
from warnings import warn

import pandas

from .execution import Executed, InlineRunner, Runner, SubprocessRunner, execute_job
from .hashing import Hash, Hasher, HashError, Location, Modules, bind_params, static_key
from .metrics import RESERVED_COLUMNS, Metric, NoActiveJob, collect, reserved
from .storage import (
    JobFolder,
    JobStore,
    PathLike,
    bind_job,
    build_manifest,
    current_job,
    drift,
    environment_fingerprint,
    witnesses_of,
)
from .tracing import DependencyTracker, TracingUnavailable, verify_called

__all__ = [
    "DependencyTracker",
    "Executed",
    "Hash",
    "HashError",
    "Hasher",
    "InlineRunner",
    "JobDecision",
    "JobFolder",
    "JobOutcome",
    "JobStore",
    "Location",
    "Metric",
    "Modules",
    "NoActiveJob",
    "Orchestrator",
    "PendingRun",
    "RunFailed",
    "RunSummary",
    "SubprocessRunner",
    "TracingUnavailable",
    "bind_job",
    "current_job",
    "execute_job",
    "static_key",
]

type RunCallable = Callable[..., None]


def owned_for(callable: Any) -> set[str]:
    """Guess the module prefixes a run's identity should be computed over.

    The user's own package and `krum` are what a change should invalidate on;
    everything else is a dependency, and belongs to the environment
    fingerprint rather than to the key.

    Args:
        callable: The user-defined function a run executes.

    Returns:
        The owned module prefixes.
    """
    module = getattr(callable, "__module__", None) or "__main__"
    return {"__main__", "krum", module.partition(".")[0]}


class PendingRun:
    """Information about an enqueued run."""

    _callable: RunCallable
    _params: dict[str, Any]
    _key: Hash | None

    __slots__ = tuple(__annotations__)

    def __init__(self, callable: RunCallable, params: dict[str, Any], key: Hash | None = None) -> None:
        """Enqueue a callable with its parameters, and optionally its key."""
        self._callable = callable
        self._params = params
        self._key = key

    def __repr__(self) -> str:
        """Render the run as its callable and parameters."""
        name = getattr(self._callable, "__qualname__", repr(self._callable))
        return f"{type(self).__qualname__}({name}, {self._params!r})"

    @property
    def callable(self) -> RunCallable:
        """The user-defined function this run executes."""
        return self._callable

    @property
    def params(self) -> dict[str, Any]:
        """The hyper parameter values this run was enqueued with."""
        return self._params

    @property
    def bound_params(self) -> dict[str, Any]:
        """The parameters in signature order, defaults included.

        This is what the key was computed from, and so what the manifest
        records: a default left unstated still identifies the job.
        """
        return bind_params(self._callable, self._params)

    @property
    def key(self) -> Hash:
        """The run's static identity, computed on first access."""
        if self._key is None:
            self._key = static_key(self._callable, self._params, owned_for(self._callable))
        return self._key


class JobDecision:
    """Whether one enqueued run needs executing, and why.

    Separating the decision from the execution is what lets a sweep be
    inspected before it is started, through :meth:`Orchestrator.plan`.
    """

    _key: str
    _action: str
    _reasons: tuple[str, ...]

    __slots__ = tuple(__annotations__)

    def __init__(self, key: str, action: str, reasons: Iterable[str] = ()) -> None:
        """Record a decision about one run."""
        self._key = key
        self._action = action
        self._reasons = tuple(reasons)

    def __repr__(self) -> str:
        """Render the decision as a constructor call."""
        return f"{type(self).__qualname__}({self._key[:12]!r}, {self._action!r}, {self._reasons!r})"

    def __str__(self) -> str:
        """Render the decision, with its reasons, on one line."""
        because = f" ({'; '.join(self._reasons)})" if self._reasons else ""
        return f"{self._action} {self._key[:16]}{because}"

    @property
    def key(self) -> str:
        """The job key."""
        return self._key

    @property
    def action(self) -> str:
        """Either `run` or `skip`."""
        return self._action

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why the job has to run; empty when it is skipped."""
        return self._reasons

    @property
    def runs(self) -> bool:
        """Whether this job would be executed."""
        return self._action == "run"


class JobOutcome:
    """What became of one enqueued run during a drain."""

    _key: str
    _status: str
    _seconds: float | None
    _path: Path | None
    _error: str | None
    _reasons: tuple[str, ...]

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        key: str,
        status: str,
        seconds: float | None = None,
        path: Path | None = None,
        error: str | None = None,
        reasons: Iterable[str] = (),
    ) -> None:
        """Record one run's outcome."""
        self._key = key
        self._status = status
        self._seconds = seconds
        self._path = path
        self._error = error
        self._reasons = tuple(reasons)

    def __repr__(self) -> str:
        """Render the outcome as a constructor call."""
        return f"{type(self).__qualname__}({self._key[:12]!r}, {self._status!r})"

    @property
    def key(self) -> str:
        """The job key."""
        return self._key

    @property
    def status(self) -> str:
        """One of `done`, `skipped` or `failed`."""
        return self._status

    @property
    def seconds(self) -> float | None:
        """How long the job took, or None if it was skipped."""
        return self._seconds

    @property
    def path(self) -> Path | None:
        """The job's folder, once it has one."""
        return self._path

    @property
    def error(self) -> str | None:
        """The traceback of a failed job."""
        return self._error

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why this job was executed rather than skipped."""
        return self._reasons


class RunSummary:
    """A report on one drain of the queue."""

    _outcomes: tuple[JobOutcome, ...]
    _pending: int

    __slots__ = tuple(__annotations__)

    def __init__(self, outcomes: Iterable[JobOutcome], pending: int = 0) -> None:
        """Summarize a drain, with the number of runs left unstarted."""
        self._outcomes = tuple(outcomes)
        self._pending = pending

    @property
    def outcomes(self) -> tuple[JobOutcome, ...]:
        """Every run's outcome, in execution order."""
        return self._outcomes

    @property
    def pending(self) -> int:
        """How many runs were left unstarted, after a failure stopped the drain."""
        return self._pending

    def count(self, status: str) -> int:
        """How many runs ended in a given status."""
        return sum(1 for outcome in self._outcomes if outcome.status == status)

    @property
    def failed(self) -> tuple[JobOutcome, ...]:
        """The runs that failed."""
        return tuple(outcome for outcome in self._outcomes if outcome.status == "failed")

    def __len__(self) -> int:
        """How many runs this drain accounted for."""
        return len(self._outcomes)

    def __repr__(self) -> str:
        """Render the summary on one line."""
        return f"{type(self).__qualname__}({self!s})"

    def __str__(self) -> str:
        """Render the per-status counts."""
        parts = [f"{self.count(status)} {status}" for status in ("done", "skipped", "failed") if self.count(status)]
        if self._pending:
            parts.append(f"{self._pending} not started")
        return ", ".join(parts) or "nothing to run"

    def report(self) -> str:
        """Render a line per run, for a human reading the end of a sweep."""
        lines = [str(self)]
        for outcome in self._outcomes:
            timing = "" if outcome.seconds is None else f"  {outcome.seconds:8.3f}s"
            because = f"  ({'; '.join(outcome.reasons)})" if outcome.reasons else ""
            lines.append(f"  {outcome.status:<8} {outcome.key[:16]}{timing}{because}")
        return "\n".join(lines)


class RunFailed(RuntimeError):
    """Raised when a run fails, stopping the sweep.

    Carries the summary of the drain, so the caller sees what ran, what was
    skipped and what was left unstarted alongside the failure itself.
    """

    _summary: RunSummary

    __slots__ = tuple(__annotations__)

    def __init__(self, summary: RunSummary, message: str) -> None:
        """Wrap a drain's summary with the failing job's message."""
        super().__init__(message)
        self._summary = summary

    @property
    def summary(self) -> RunSummary:
        """The summary of the drain this failure stopped."""
        return self._summary


class Orchestrator:
    """Top-most orchestrator instance managing runs and persisting metrics.

    Args:
        root: The directory holding one folder per job.
        owned: What a job's identity is computed over, as module prefixes or
            as a :class:`~krum.orchestration.hashing.Modules` carrying its own exclusions; guessed per run
            from the callable's own package by default.
        lock: The lock file fingerprinting the environment; discovered from the
            current directory by default.
        source: A directory inside the repository holding the code being run,
            whose commit is recorded with every job; the current directory by
            default.
        force: Re-run every job, whatever is already recorded.
        isolate: Run each job in a fresh interpreter of its own, which keeps
            one job's leftover state from reaching the next. This requires the
            experiment and its parameters to be picklable, and a sweep
            script's top level to be guarded by `if __name__ == "__main__":`;
            see :mod:`krum.orchestration.execution`.
        trace: Record which owned functions each job actually entered, and
            re-check them on a later pass. This closes the gap a static read of
            the code leaves open, a dependency reached only at runtime being
            invisible to a job's key; see :mod:`krum.orchestration.tracing`.
            On by default: the cost is one callback per distinct function, and
            the alternative is keeping a stale result. Turn it off for a sweep
            that must share monitoring with a debugger or a profiler.
    """

    _store: JobStore
    _queue: Deque[PendingRun]
    _enqueued: list[str]
    _owned: Modules | Iterable[str] | None
    _lock: PathLike | None
    _source: PathLike | None
    _force: bool
    _isolate: bool
    _trace: bool
    _witnesses: dict[str, Any] | None
    _warned: bool

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        root: PathLike,
        *,
        owned: Modules | Iterable[str] | None = None,
        lock: PathLike | None = None,
        source: PathLike | None = None,
        force: bool = False,
        isolate: bool = False,
        trace: bool = True,
    ) -> None:
        """Open a store at `root`, creating it if needed."""
        self._store = JobStore(root)
        self._queue = Deque()
        self._enqueued = []
        self._owned = owned
        self._lock = lock
        self._source = source
        self._force = force
        self._isolate = isolate
        self._trace = trace
        self._witnesses = None
        self._warned = False

    def __repr__(self) -> str:
        """Render the orchestrator with its root and queue depth."""
        return f"{type(self).__qualname__}({str(self._store.root)!r}, queued={len(self._queue)})"

    def __enter__(self) -> Self:
        """Enter a sweep; the queue is drained on a clean exit."""
        return self

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, tb: TracebackType | None) -> None:
        """Drain the queue, unless the block is already unwinding."""
        if exc_type is None:
            self.drain()

    @property
    def store(self) -> JobStore:
        """The job store this orchestrator reads and writes."""
        return self._store

    @property
    def queued(self) -> int:
        """How many runs are enqueued."""
        return len(self._queue)

    def _owned_for(self, callable: RunCallable) -> Modules | Iterable[str]:
        """What a run's identity is computed over, as given."""
        return owned_for(callable) if self._owned is None else self._owned

    def _prefixes_for(self, callable: RunCallable) -> tuple[str, ...]:
        """The owned module prefixes for one run, as plain names.

        An ownership test carries exclusions that plain prefixes cannot, so it
        is accepted as given and unwrapped here, where the manifest and the
        tracer both want names.
        """
        owned = self._owned_for(callable)
        return tuple(sorted(owned.prefixes if isinstance(owned, Modules) else owned))

    def run(self, callable: RunCallable, **params: Any) -> Hash:
        """Enqueue one run, computing its identity now.

        The key is computed eagerly so that an ill-fitting parameter, or a
        dependency that cannot be hashed, is reported at the call site rather
        than after a long sweep has already started.

        Args:
            callable: The user-defined function to execute.
            **params: The hyper parameter values to execute it on.

        Returns:
            The run's key, which names its output folder.

        Raises:
            TypeError: If the parameters do not fit the callable's signature.
            ValueError: If a parameter name is one a metric frame owns.
        """
        clashing = reserved(bind_params(callable, params))
        if clashing:
            raise ValueError(
                f"parameter names {clashing} are reserved: a metric frame's own columns are "
                f"{list(RESERVED_COLUMNS)}, so such a parameter could not be told apart from "
                "the metric it was recorded against; rename it"
            )
        key = static_key(callable, params, self._owned_for(callable))
        self._queue.append(PendingRun(callable, params, key))
        name = self._store.name_for(key)
        if name not in self._enqueued:
            self._enqueued.append(name)
        return key

    def runner(self, owned: Iterable[str] = ()) -> Runner:
        """How this orchestrator executes a job's body.

        Args:
            owned: Module prefixes whose functions are worth recording, when
                tracing. A sweep is normally one experiment, so this is taken
                from the first enqueued run.

        Returns:
            The runner, configured for isolation and tracing.
        """
        prefixes = tuple(owned)
        if self._isolate:
            return SubprocessRunner(prefixes, self._trace)
        return InlineRunner(prefixes, self._trace)

    def _owned_prefixes(self) -> tuple[str, ...]:
        """The owned prefixes a traced sweep records against."""
        if not self._queue:
            return ()
        return self._prefixes_for(self._queue[0].callable)

    def witnesses(self) -> dict[str, Any]:
        """The facts the current environment would stamp on a result.

        Computed once per orchestrator: a sweep is one environment, and the
        lock file should not be re-read for every job.
        """
        if self._witnesses is None:
            self._witnesses = witnesses_of(environment_fingerprint(self._lock, self._source))
        return self._witnesses

    def decide(self, pending: PendingRun, force: bool | None = None) -> JobDecision:
        """Decide whether one enqueued run needs executing, and say why.

        A job runs when it has no recorded result, when its last attempt
        failed, when the environment it was recorded in no longer matches, or
        when it is forced. Otherwise its stored output stands.

        Args:
            pending: The enqueued run to decide on.
            force: Override the orchestrator's own `force` setting.

        Returns:
            The decision, carrying the reasons a job has to run.
        """
        key = self._store.name_for(pending.key)
        folder = self._store.folder_for(key)
        status = folder.status
        if status == "absent":
            return JobDecision(key, "run", ("no recorded result",))
        if status == "failed":
            return JobDecision(key, "run", ("previous attempt failed",))
        if self._force if force is None else force:
            return JobDecision(key, "run", ("forced",))
        reasons = drift(folder.witnesses(), self.witnesses())
        reasons += verify_called(folder.called() or {})
        return JobDecision(key, "run" if reasons else "skip", reasons)

    def plan(self, force: bool | None = None) -> list[JobDecision]:
        """What a drain would do, without executing anything.

        Args:
            force: Override the orchestrator's own `force` setting.

        Returns:
            One decision per enqueued run, in queue order.
        """
        return [self.decide(pending, force) for pending in self._queue]

    def drain(self, force: bool | None = None) -> RunSummary:
        """Execute every enqueued run that is not already done.

        A run whose folder is marked done is skipped. A run that fails stops
        the drain, and the queue is abandoned: the results that did complete
        stay readable, which they would not be if a later `get` re-ran the
        failing job. Re-enqueueing retries it, its folder being marked failed
        rather than done.

        Args:
            force: Override the orchestrator's own `force` setting.

        Returns:
            The summary of this drain.

        Raises:
            RunFailed: If a run raised, after recording its traceback.
        """
        with self.runner(self._owned_prefixes()) as runner:
            return self._drain(runner, force)

    def _drain(self, runner: Runner, force: bool | None) -> RunSummary:
        """Execute the queue with a prepared runner."""
        outcomes: list[JobOutcome] = []
        while self._queue:
            pending = self._queue[0]
            decision = self.decide(pending, force)
            key = decision.key
            if not decision.runs:
                self._queue.popleft()
                outcomes.append(JobOutcome(key, "skipped", path=self._store.folder_for(key).path))
                continue
            manifest = build_manifest(
                key,
                pending.callable,
                pending.bound_params,
                owned=self._prefixes_for(pending.callable),
                lock=self._lock,
                start=self._source,
            )
            self._warn_if_dirty(manifest)
            writer = self._store.writer(key, manifest)
            started = perf_counter()
            try:
                executed = runner.run(writer, pending.callable, pending.params)
            except BaseException:
                # Interrupted rather than failed: leave no marker, so that the
                # next drain reads this job as "not done" and runs it again.
                writer.discard()
                raise
            writer.record_called(executed.called)
            if not executed.ok:
                report = executed.error or "the job failed without a traceback"
                failed = writer.finish("failed", report)
                outcomes.append(
                    JobOutcome(key, "failed", perf_counter() - started, failed.path, report, decision.reasons)
                )
                summary = RunSummary(outcomes, pending=len(self._queue) - 1)
                self._queue.clear()
                raise RunFailed(summary, f"job {key[:16]} failed\n{report}\n{summary.report()}") from executed.exception
            done = writer.finish("done")
            self._queue.popleft()
            outcomes.append(JobOutcome(key, "done", perf_counter() - started, done.path, reasons=decision.reasons))
        return RunSummary(outcomes)

    def _warn_if_dirty(self, manifest: Mapping[str, Any]) -> None:
        """Warn once if results are being produced from a modified working tree."""
        if self._warned:
            return
        git = manifest.get("git")
        if git is not None and git.get("dirty"):
            self._warned = True
            warn(
                f"recording results from a dirty working tree at {git.get('commit', '?')[:12]}; "
                "the stored commit will not reproduce them",
                stacklevel=3,
            )

    @property
    def enqueued(self) -> tuple[str, ...]:
        """The jobs of this sweep, in the order they were enqueued."""
        return tuple(self._enqueued)

    def get(self, metric: str) -> pandas.DataFrame:
        """Read one metric back across this sweep's jobs.

        Drains the queue first, so that a sweep need not be run explicitly.

        Only the jobs enqueued here are read, in that order. A store
        accumulates a folder per version of an experiment, so reading all of
        them would mix code versions, two of which can carry the same label
        and be drawn over each other. Pass
        :func:`krum.orchestration.metrics.collect` the store directly to read
        everything it holds.

        Args:
            metric: The metric name.

        Returns:
            A tidy frame of `[step, value, *params, job_key]`.

        Raises:
            KeyError: If none of this sweep's jobs recorded that metric.
            RunFailed: If draining the queue hit a failing run.
        """
        self.drain()
        return collect(self._store, metric, self._enqueued or None)

    def metrics(self) -> list[str]:
        """Every metric name recorded by a completed job, sorted."""
        return self._store.metric_names()
