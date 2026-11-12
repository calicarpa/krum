"""Demonstrate the live progress display of the orchestrator.

Run from the repository root:

    uv run python examples/progress_demo.py

It drains a short sweep four times. The first pass runs every job, the second
reuses the cache, the third stops on a job that fails, and the fourth retries
that job and counts it as a rerun. With ``rich`` installed the display is a
rich bar; without it, a plain ANSI line.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from krum.orchestration import Metric, Orchestrator, RunFailed


def slow_job(seed: int, work: float) -> None:
    """Pretend to train: sleep for ``work`` seconds, then record a metric."""
    time.sleep(work)
    Metric("loss", dtype=float).push(seed, float(seed) / 10.0)


def flaky_job(seed: int, work: float, marker: str) -> None:
    """Fail while ``marker`` exists, then behave like ``slow_job``.

    The failure is driven by a file rather than by a change to this function,
    so both attempts share the same job key: editing the function would change
    its identity and turn the second attempt into a new job instead of a retry.
    """
    flag = Path(marker)
    if flag.exists():
        flag.unlink()
        raise RuntimeError("transient failure, retry this job")
    slow_job(seed, work)


def enqueue(
    orchestrator: Orchestrator,
    work: list[float],
    *,
    start: int = 0,
    marker: Path | None = None,
    fail_at: int | None = None,
) -> None:
    """Enqueue one job per duration, making the ``fail_at`` seed the flaky one."""
    for offset, duration in enumerate(work):
        seed = start + offset
        if marker is not None and seed == fail_at:
            orchestrator.run(flaky_job, seed=seed, work=duration, marker=str(marker))
        else:
            orchestrator.run(slow_job, seed=seed, work=duration)


def main() -> None:
    """Run the demo: a fresh drain, a cached one, then a failure and its retry."""
    store = Path(tempfile.mkdtemp(prefix="krum-progress-demo-"))
    work = [0.04 + 0.04 * (i % 5) for i in range(60)]
    fresh = [0.04 + 0.04 * (i % 5) for i in range(5)]

    print("first pass: running every job")
    orchestrator = Orchestrator(store)
    enqueue(orchestrator, work)
    orchestrator.drain()

    print(f"second pass: reusing the cache in {store}")
    orchestrator = Orchestrator(store)
    enqueue(orchestrator, work)
    orchestrator.drain()

    marker = Path(tempfile.mkdtemp(prefix="krum-progress-demo-failure-")) / "fail-once"
    marker.touch()
    print("third pass: stopping on the third new job")
    orchestrator = Orchestrator(store)
    enqueue(orchestrator, fresh, start=60, marker=marker, fail_at=62)
    try:
        orchestrator.drain()
    except RunFailed as failed:
        outcome = failed.summary.failed[0]
        print(f"drain stopped on {outcome.key[:12]} and left {failed.summary.pending} jobs unstarted")

    print("fourth pass: retrying the failed job")
    orchestrator = Orchestrator(store)
    enqueue(orchestrator, fresh, start=60, marker=marker, fail_at=62)
    orchestrator.drain()


if __name__ == "__main__":
    main()
