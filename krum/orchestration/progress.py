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
from time import perf_counter
from typing import Protocol, TextIO

__all__ = ["NullProgress", "Progress", "make_progress"]


class Progress(Protocol):
    """A live progress display for a single drain."""

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

    def advance(self, *, ran: bool, duration: float = 0.0, description: str = "") -> None:
        """Record one completed job without rendering anything."""

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
        line = (
            f"\r\x1b[K{_bar(self.completed, self.total)} "
            f"{self.completed}/{self.total} · elapsed {_fmt_seconds(self.elapsed())}"
            f" · ETA {_fmt_seconds(self.eta())}"
        )
        if description:
            line += f" · {description}"
        self._stream.write(line)
        self._stream.flush()

    def close(self) -> None:
        """Move to a fresh line so later output does not overwrite the bar."""
        self._stream.write("\n")
        self._stream.flush()


class RichProgress(_Tracker):
    """A rich-based bar with a spinner, percentage and elapsed column."""

    def __init__(self, total: int, runs_total: int, *, stream: TextIO) -> None:
        super().__init__(total, runs_total)
        from rich.console import Console
        from rich.progress import (
            BarColumn,
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
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=Console(file=stream),
        )
        self._task = self._rich.add_task("sweep", total=self.total)
        self._rich.start()

    def render(self, description: str) -> None:
        """Refresh the rich display with the current ETA."""
        label = f"ETA {_fmt_seconds(self.eta())}"
        if description:
            label += f" · {description}"
        self._rich.update(self._task, completed=self.completed, description=label)

    def close(self) -> None:
        """Stop the rich display and restore the terminal."""
        self._rich.stop()


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
