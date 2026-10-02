"""How a job's body is executed: in this process, or in one of its own.

Running each job in a fresh interpreter is a reproducibility measure before it
is a performance one. Nothing a job leaves behind — a seeded global generator,
an imported module's mutable state, an initialised accelerator context — can
reach the next job, so a sweep's results do not depend on the order its jobs
happened to run in.

It costs two requirements, which is why it is opted into rather than assumed:

- The experiment and its parameters must be picklable, since they are sent to
  the child. A function or class is pickled by name, so it has to be reachable
  at module level rather than defined inside another function.
- A sweep script's top level must be guarded by `if __name__ == "__main__":`.
  A spawned child re-imports the module it came from, and an unguarded sweep
  would start itself again there.

See `notes/2026-10-02-orchestrator-v2-a-design.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from multiprocessing import get_context
from traceback import format_exc
from types import TracebackType
from typing import Any, Self

from .storage import JobWriter, MetricRecorder, PathLike, bind_job
from .tracing import DependencyTracker


class Executed:
    """The outcome of running one job's body."""

    _error: str | None
    _exception: BaseException | None
    _called: dict[str, str] | None

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        error: str | None = None,
        exception: BaseException | None = None,
        called: dict[str, str] | None = None,
    ) -> None:
        """Record a body's outcome, with the traceback if it failed.

        Args:
            error: The formatted traceback.
            exception: The exception itself, when the job ran in this process.
                A job run in a child leaves only its traceback behind, so this
                is None there, and the orchestrator chains onto it only when
                there is something to chain onto.
            called: The functions the job entered, when it was traced. None
                means the job was not traced, which is not the same as a job
                that was traced and called nothing.
        """
        self._error = error
        self._exception = exception
        self._called = called

    def __repr__(self) -> str:
        """Render the outcome as a constructor call."""
        return f"{type(self).__qualname__}({'failed' if self._error else 'ok'!r})"

    @property
    def ok(self) -> bool:
        """Whether the body ran to completion."""
        return self._error is None

    @property
    def error(self) -> str | None:
        """The traceback, if the body raised."""
        return self._error

    @property
    def exception(self) -> BaseException | None:
        """The exception itself, for a job that ran in this process."""
        return self._exception

    @property
    def called(self) -> dict[str, str] | None:
        """The functions the job entered, or None if it was not traced."""
        return self._called


def execute_job(
    path: PathLike,
    callable: Callable[..., Any],
    params: Mapping[str, Any],
    owned: tuple[str, ...] = (),
    trace: bool = False,
) -> dict[str, Any]:
    """Run one job's body, recording its metrics into an already staged directory.

    This is the child process entry point, so it is a module-level function
    that returns rather than raises: an exception's traceback does not survive
    being sent back to the parent, so the traceback is formatted here, where
    the frames still exist.

    Args:
        path: The staged directory to record into.
        callable: The user-defined function to execute.
        params: The parameters to execute it on.
        owned: Module prefixes whose functions are worth recording.
        trace: Record which owned functions the job entered.

    Returns:
        The formatted traceback if any, the metrics that were registered, and
        the functions entered when tracing was asked for.
    """
    recorder = MetricRecorder(path)
    error = None
    tracker = DependencyTracker(owned) if trace else None
    try:
        with ExitStack() as stack:
            if tracker is not None:
                stack.enter_context(tracker)
            stack.enter_context(bind_job(recorder))
            callable(**params)
    except Exception:
        error = format_exc()
    return {
        "error": error,
        "metrics": recorder.close(),
        "called": None if tracker is None else tracker.called(),
    }


class InlineRunner:
    """Runs each job in the orchestrator's own process.

    Args:
        owned: Module prefixes whose functions are worth recording.
        trace: Record which owned functions each job entered.
    """

    _owned: tuple[str, ...]
    _trace: bool

    __slots__ = tuple(__annotations__)

    def __init__(self, owned: Iterable[str] = (), trace: bool = False) -> None:
        """Prepare to run jobs in this process."""
        self._owned = tuple(owned)
        self._trace = trace

    def __repr__(self) -> str:
        """Render the runner as a constructor call."""
        return f"{type(self).__qualname__}({self._owned!r}, trace={self._trace})"

    def run(self, writer: JobWriter, callable: Callable[..., Any], params: Mapping[str, Any]) -> Executed:
        """Execute a job's body here and now.

        An interrupt is left to propagate, so that the orchestrator can discard
        the staged directory rather than promote a marker for a job that never
        finished.
        """
        tracker = DependencyTracker(self._owned) if self._trace else None
        try:
            with ExitStack() as stack:
                if tracker is not None:
                    stack.enter_context(tracker)
                stack.enter_context(bind_job(writer))
                callable(**params)
        except Exception as error:
            return Executed(format_exc(), error, None if tracker is None else tracker.called())
        return Executed(called=None if tracker is None else tracker.called())

    def close(self) -> None:
        """Nothing to release."""

    def __enter__(self) -> Self:
        """Enter a drain."""
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Release whatever the runner holds."""
        self.close()


class SubprocessRunner:
    """Runs each job in a fresh interpreter of its own.

    The pool is spawned rather than forked, and retires each worker after one
    job, so no two jobs share an interpreter. Raising `workers` above one is
    all that stands between this and running a sweep in parallel; it is left at
    one for now, the orchestrator still being fail-fast and sequential.
    """

    _owned: tuple[str, ...]
    _trace: bool
    _workers: int
    _executor: ProcessPoolExecutor | None

    __slots__ = tuple(__annotations__)

    def __init__(self, owned: Iterable[str] = (), trace: bool = False, workers: int = 1) -> None:
        """Prepare to run jobs out of process, without starting anything yet."""
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self._owned = tuple(owned)
        self._trace = trace
        self._workers = workers
        self._executor = None

    def __repr__(self) -> str:
        """Render the runner as a constructor call."""
        return f"{type(self).__qualname__}({self._owned!r}, trace={self._trace}, workers={self._workers})"

    @property
    def workers(self) -> int:
        """How many jobs may run at once."""
        return self._workers

    def executor(self) -> ProcessPoolExecutor:
        """The pool, started on first use."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=get_context("spawn"),
                max_tasks_per_child=1,
            )
        return self._executor

    def run(self, writer: JobWriter, callable: Callable[..., Any], params: Mapping[str, Any]) -> Executed:
        """Execute a job's body in a child process and collect what it recorded.

        A child that dies outright, rather than raising, leaves the pool
        unusable; it is discarded so that the next job starts from a fresh one,
        and the death is reported as that job failing.
        """
        try:
            submitted = self.executor().submit(
                execute_job, str(writer.path), callable, params, self._owned, self._trace
            )
            result = submitted.result()
        except BrokenProcessPool:
            self.close()
            return Executed(f"{format_exc()}\nthe child process running this job died")
        except Exception as error:
            # The job never started: its body or its parameters could not be
            # sent to the child. Every job would hit this, so it stops the
            # sweep rather than being recorded as one job failing.
            self.close()
            raise RuntimeError(
                "could not send this job to a child process, so it cannot run in isolation; "
                "the experiment and every parameter must be picklable, which means reachable "
                "at module level rather than defined inside another function"
            ) from error
        writer.adopt(result["metrics"])
        return Executed(result["error"], called=result["called"])

    def close(self) -> None:
        """Shut the pool down, abandoning anything still queued."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> Self:
        """Enter a drain."""
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Shut the pool down."""
        self.close()


type Runner = InlineRunner | SubprocessRunner
