"""Live progress display for orchestrator drains.

A drain runs a queue of jobs. On an interactive terminal it can show how far
the sweep has got, the elapsed time and an estimated time remaining. The
estimate is based on the jobs that actually executed, so a sweep that mostly
reuses cached results does not overstate how long is left.

The display degrades gracefully. ``rich`` is used when it is importable, a
plain ANSI line otherwise, and nothing at all when the output is not a
terminal. There is no hard dependency on any third-party package.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TextIO

__all__ = ["Failure", "NullProgress", "Progress", "describe_params", "make_progress"]


def describe_params(params: Mapping[str, Any], limit: int | None = 80) -> str:
    """Render a job's parameters as a ``k=v · k=v`` string.

    Class values are shortened to their qualified name, and the whole string
    is capped so a wide sweep does not fill the terminal. Pass ``limit=None``
    to keep every parameter, which is what a failure report does.
    """
    parts: list[str] = []
    for name, value in params.items():
        if isinstance(value, type):
            value = value.__qualname__
        parts.append(f"{name}={value}")
    text = " · ".join(parts)
    if limit is not None and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text or "(no params)"


@dataclass(frozen=True, slots=True)
class Failure:
    """What a failed job needs to be diagnosed from the console.

    Carries the job's identity, the parameters it ran with, the folder its
    traceback was written to, and the traceback itself.
    """

    key: str
    params: Mapping[str, Any]
    error: str
    path: Path | None = None


class Progress(Protocol):
    """A live progress display for a single drain."""

    def describe(self, description: str = "") -> None:
        """Show a job about to start, without advancing the counter."""

    def advance(self, *, ran: bool, rerun: bool = False, duration: float = 0.0, description: str = "") -> None:
        """Record one completed job.

        Args:
            ran: Whether the job actually executed, as opposed to being reused
                from the cache.
            rerun: Whether the job is being retried after a previous failed
                attempt.
            duration: How long the job took to execute, in seconds. Ignored
                when ``ran`` is false.
            description: A short label for the job just finished.
        """

    def fail(self, failure: Failure) -> None:
        """Record the job that stopped the drain."""

    def close(self) -> None:
        """Finish the display, leaving the terminal in a clean state."""


class NullProgress:
    """A progress display that renders nothing."""

    def describe(self, description: str = "") -> None:
        """No-op."""

    def advance(self, *, ran: bool, rerun: bool = False, duration: float = 0.0, description: str = "") -> None:
        """No-op."""

    def fail(self, failure: Failure) -> None:
        """No-op."""

    def close(self) -> None:
        """No-op."""


def _fmt_seconds(seconds: float | None) -> str:
    """Render seconds as ``MM:SS``, or a placeholder when unknown."""
    if seconds is None:
        return "--:--"
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _bar(completed: int, total: int, width: int = 20) -> str:
    """An ASCII progress bar ``[#####-----]``."""
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = round(width * completed / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class _Tracker:
    """Shared accounting: completion, run durations and the time left."""

    def __init__(self, total: int, runs_total: int) -> None:
        self.total = total
        self.runs_total = runs_total
        self.completed = 0
        self.runs_done = 0
        self.reruns = 0
        self.failed = 0
        self.failure: Failure | None = None
        self.run_time = 0.0
        self._start = perf_counter()

    def describe(self, description: str = "") -> None:
        """Refresh the display with a job about to run."""
        self.render(description)

    def advance(self, *, ran: bool, rerun: bool = False, duration: float = 0.0, description: str = "") -> None:
        """Record one completed job and refresh the display."""
        self.completed += 1
        if ran:
            self.runs_done += 1
            self.run_time += duration
        if rerun:
            self.reruns += 1
        self.render(description)

    def fail(self, failure: Failure) -> None:
        """Record the job that stopped the drain and refresh the display."""
        self.failed += 1
        self.failure = failure
        self.render("")

    def elapsed(self) -> float:
        """Seconds since the display was created."""
        return perf_counter() - self._start

    def eta(self) -> float | None:
        """Estimated seconds until the last run completes."""
        if self.runs_done == 0:
            return None
        average = self.run_time / self.runs_done
        return average * (self.runs_total - self.runs_done)

    def rate(self) -> float:
        """Jobs completed per second, or zero before the first."""
        elapsed = self.elapsed()
        return self.completed / elapsed if elapsed > 0 else 0.0

    @property
    def cached(self) -> int:
        """Jobs reused from the cache."""
        return self.completed - self.runs_done

    def statuses(self) -> list[tuple[str, int]]:
        """Non-zero status counts as ``(label, count)`` pairs."""
        statuses = [("run", self.runs_done)]
        if self.reruns:
            statuses.append(("rerun", self.reruns))
        if self.cached:
            statuses.append(("cached", self.cached))
        if self.failed:
            statuses.append(("failed", self.failed))
        return statuses

    def failure_lines(self) -> list[str]:
        """The failing job's identity, parameters and traceback, as plain lines."""
        failure = self.failure
        if failure is None:
            return []
        lines = [f"job {failure.key} failed"]
        if failure.path is not None:
            lines.append(f"  folder {failure.path}")
        lines.append(f"  params {describe_params(failure.params, limit=None)}")
        lines.extend(f"  {line}" for line in failure.error.rstrip().splitlines())
        return lines

    def render(self, description: str) -> None:
        """Draw the current state; implemented by each renderer."""
        raise NotImplementedError


class PlainProgress(_Tracker):
    """A single ANSI line, cleared and rewritten on every advance."""

    _CODES = {"run": "36", "rerun": "33", "cached": "2", "failed": "31"}

    def __init__(self, total: int, runs_total: int, *, stream: TextIO) -> None:
        super().__init__(total, runs_total)
        self._stream = stream

    def _colored(self, label: str, count: int) -> str:
        """Color a status count for an ANSI terminal."""
        return f"\x1b[{self._CODES[label]}m{count} {label}\x1b[0m"

    def render(self, description: str) -> None:
        """Draw one progress line."""
        counts = " · ".join(self._colored(label, count) for label, count in self.statuses())
        line = f"\r\x1b[K{_bar(self.completed, self.total)} {self.completed}/{self.total} · {counts}"
        line += f" · elapsed {_fmt_seconds(self.elapsed())} · ETA {_fmt_seconds(self.eta())} · {self.rate():.1f} job/s"
        if description:
            line += f" · {description}"
        self._stream.write(line)
        self._stream.flush()

    def close(self) -> None:
        """Replace the bar with the failure report and the summary."""
        lines: list[str] = []
        if self.failure is not None:
            lines = self.failure_lines()
            lines[0] = f"\x1b[{self._CODES['failed']}m{lines[0]}\x1b[0m"
            lines.append("")
        lines.extend([
            "Sweep complete",
            f"  {self._colored('run', self.runs_done)}",
        ])
        if self.reruns:
            lines.append(f"  {self._colored('rerun', self.reruns)}")
        lines.append(f"  {self._colored('cached', self.cached)}")
        lines.append(f"  {self._colored('failed', self.failed)}")
        lines.append(f"  elapsed {_fmt_seconds(self.elapsed())} · {self.rate():.1f} job/s")
        self._stream.write("\r\x1b[K" + "\n".join(lines) + "\n")
        self._stream.flush()


class RichProgress(_Tracker):
    """A rich-based bar with counts, ETA, throughput and the current job."""

    _COLORS = {"run": "cyan", "rerun": "yellow", "cached": "dim", "failed": "red"}

    def __init__(self, total: int, runs_total: int, *, stream: TextIO) -> None:
        super().__init__(total, runs_total)
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.progress import (
            Progress as Rich,
        )

        self._rich = Rich(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[counts]}"),
            TimeElapsedColumn(),
            TextColumn("ETA {task.fields[eta]}"),
            TextColumn("{task.fields[rate]}"),
            console=Console(file=stream),
            # The bar owns the status stream only: a job that prints must keep
            # writing to the stdout or stderr it would have written to anyway,
            # rather than having `sys.stdout` swapped for the duration of a drain.
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._task = self._rich.add_task(
            "sweep",
            total=self.total,
            counts="0 run",
            eta="--:--",
            rate="0.0 job/s",
        )
        self._rich.start()

    def _counts(self) -> str:
        """The status breakdown as colored rich markup."""
        return " · ".join(f"[{self._COLORS[label]}]{count} {label}[/]" for label, count in self.statuses())

    def render(self, description: str) -> None:
        """Refresh the rich display with the current state."""
        self._rich.update(
            self._task,
            completed=self.completed,
            description=description,
            counts=self._counts(),
            eta=_fmt_seconds(self.eta()),
            rate=f"{self.rate():.1f} job/s",
        )

    def close(self) -> None:
        """Stop the bar, report the failure if there was one, then summarize."""
        self._rich.stop()
        if self.failure is not None:
            self._print_failure()
        self._print_summary()

    def _print_failure(self) -> None:
        """Render the failing job as a red panel, with its traceback folded in."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        failure = self.failure
        if failure is None:
            return
        details = Table.grid(padding=(0, 2))
        details.add_column(style="dim", no_wrap=True)
        details.add_column(overflow="fold")
        details.add_row("key", Text(failure.key))
        if failure.path is not None:
            details.add_row("folder", Text(str(failure.path)))
        details.add_row("params", Text(describe_params(failure.params, limit=None)))
        error = failure.error.rstrip().splitlines() or [""]
        body = Group(
            details,
            Text(""),
            Text("\n".join(error[:-1]), style="red"),
            Text(error[-1], style="bold red"),
        )
        self._rich.console.print(
            Panel(body, title="[bold red]Job failed[/]", title_align="left", border_style="red", padding=(0, 1))
        )

    def _print_summary(self) -> None:
        """Print the counts, elapsed time and throughput of the whole drain."""
        from rich.table import Table

        table = Table(
            title="Sweep complete", title_justify="left", show_header=False, border_style="dim", padding=(0, 1)
        )
        table.add_column(style="dim", no_wrap=True)
        table.add_column(justify="right")
        table.add_row("[cyan]Run[/]", f"[cyan]{self.runs_done}[/]")
        if self.reruns:
            table.add_row("[yellow]Rerun[/]", f"[yellow]{self.reruns}[/]")
        table.add_row("[dim]Cached[/]", f"[dim]{self.cached}[/]")
        table.add_row("[red]Failed[/]", f"[red]{self.failed}[/]")
        table.add_row("Elapsed", _fmt_seconds(self.elapsed()))
        table.add_row("Throughput", f"{self.rate():.1f} job/s")
        self._rich.console.print(table)


def make_progress(
    total: int,
    runs_total: int,
    *,
    enabled: bool = True,
    stream: TextIO | None = None,
) -> Progress:
    """Build the best available progress display for a drain of ``total`` jobs.

    Args:
        total: The number of jobs in the drain.
        runs_total: How many of them will actually execute, the rest being
            reused from the cache.
        enabled: When false, return a no-op display.
        stream: The stream to draw on; standard error by default.

    Returns:
        A no-op on a non-terminal stream or when disabled, a rich bar when
        ``rich`` is importable, and a plain ANSI line otherwise.
    """
    if not enabled:
        return NullProgress()
    stream = stream if stream is not None else sys.stderr
    if not stream.isatty():
        return NullProgress()
    try:
        return RichProgress(total, runs_total, stream=stream)
    except ImportError:
        return PlainProgress(total, runs_total, stream=stream)
