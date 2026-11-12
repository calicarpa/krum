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
from time import perf_counter
from typing import Any, Protocol, TextIO

__all__ = ["NullProgress", "Progress", "describe_params", "make_progress"]


def describe_params(params: Mapping[str, Any], limit: int = 80) -> str:
    """Render a job's parameters as a short ``k=v · k=v`` string.

    Class values are shortened to their qualified name, and the whole string
    is capped so a wide sweep does not fill the terminal.
    """
    parts: list[str] = []
    for name, value in params.items():
        if isinstance(value, type):
            value = value.__qualname__
        parts.append(f"{name}={value}")
    text = " · ".join(parts)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text or "(no params)"


class Progress(Protocol):
    """A live progress display for a single drain."""

    def describe(self, description: str = "") -> None:
        """Show a job about to start, without advancing the counter."""

    def advance(self, *, ran: bool, duration: float = 0.0, description: str = "") -> None:
        """Record one completed job.

        Args:
            ran: Whether the job actually executed, as opposed to being reused
                from the cache.
            duration: How long the job took to execute, in seconds. Ignored
                when ``ran`` is false.
            description: A short label for the job just finished.
        """

    def close(self) -> None:
        """Finish the display, leaving the terminal in a clean state."""


class NullProgress:
    """A progress display that renders nothing."""

    def describe(self, description: str = "") -> None:
        """No-op."""

    def advance(self, *, ran: bool, duration: float = 0.0, description: str = "") -> None:
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
        self.run_time = 0.0
        self._start = perf_counter()

    def describe(self, description: str = "") -> None:
        """Refresh the display with a job about to run."""
        self.render(description)

    def advance(self, *, ran: bool, duration: float = 0.0, description: str = "") -> None:
        """Record one completed job and refresh the display."""
        self.completed += 1
        if ran:
            self.runs_done += 1
            self.run_time += duration
        self.render(description)

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

    def summary(self) -> str:
        """A one-line summary of the whole drain."""
        return (
            f"{self.completed} jobs · {self.runs_done} run · {self.cached} cached · "
            f"{_fmt_seconds(self.elapsed())} · {self.rate():.1f} job/s"
        )

    def render(self, description: str) -> None:
        """Draw the current state; implemented by each renderer."""
        raise NotImplementedError


class PlainProgress(_Tracker):
    """A single ANSI line, cleared and rewritten on every advance."""

    def __init__(self, total: int, runs_total: int, *, stream: TextIO) -> None:
        super().__init__(total, runs_total)
        self._stream = stream

    def render(self, description: str) -> None:
        """Draw one progress line."""
        line = f"\r\x1b[K{_bar(self.completed, self.total)} {self.completed}/{self.total} · {self.runs_done} run"
        if self.cached:
            line += f" · {self.cached} cached"
        line += f" · elapsed {_fmt_seconds(self.elapsed())} · ETA {_fmt_seconds(self.eta())} · {self.rate():.1f} job/s"
        if description:
            line += f" · {description}"
        self._stream.write(line)
        self._stream.flush()

    def close(self) -> None:
        """Replace the bar with the summary and move to a fresh line."""
        self._stream.write(f"\r\x1b[K{self.summary()}\n")
        self._stream.flush()


class RichProgress(_Tracker):
    """A rich-based bar with counts, ETA, throughput and the current job."""

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
        """The ``N run · M cached`` breakdown, without the cached part at zero."""
        parts = [f"{self.runs_done} run"]
        if self.cached:
            parts.append(f"{self.cached} cached")
        return " · ".join(parts)

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
        """Stop the bar and print the summary."""
        self._rich.stop()
        self._rich.console.print(f"[bold green]✓[/] {self.summary()}")


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
