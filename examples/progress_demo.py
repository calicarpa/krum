"""Demonstrate the live progress display of the orchestrator.

Run from the repository root:

    uv run python examples/progress_demo.py

It enqueues a sweep of short jobs, drains it once (running every job), then
drains it again (reusing the cache). The first pass shows the progress bar
advancing and the ETA updating; the second pass shows the cached jobs being
skipped in a burst. With ``rich`` installed, the display is a rich bar; without
it, a plain ANSI line.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from krum.orchestration import Metric, Orchestrator


def slow_job(seed: int, work: float) -> None:
    """Pretend to train: sleep for ``work`` seconds, then record a metric."""
    time.sleep(work)
    Metric("loss", dtype=float).push(seed, float(seed) / 10.0)


def main() -> None:
    """Run the demo: one fresh drain, then one fully cached drain."""
    store = Path(tempfile.mkdtemp(prefix="krum-progress-demo-"))
    work = [0.04 + 0.04 * (i % 5) for i in range(60)]

    print("first pass: running every job")
    orchestrator = Orchestrator(store)
    for seed, duration in enumerate(work):
        orchestrator.run(slow_job, seed=seed, work=duration)
    orchestrator.drain()

    print(f"second pass: reusing the cache in {store}")
    orchestrator = Orchestrator(store)
    for seed, duration in enumerate(work):
        orchestrator.run(slow_job, seed=seed, work=duration)
    orchestrator.drain()


if __name__ == "__main__":
    main()
