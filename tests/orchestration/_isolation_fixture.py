"""Experiments for the isolation tests, at module level so a child can import them.

A spawned child receives an experiment by name and imports it, so these cannot
live inside a test method: a function defined there is not picklable, which is
the very constraint :mod:`krum.orchestration.execution` documents.
"""

import os

from krum.orchestration import Metric

# Appended to by every job. In a shared interpreter the second job sees what
# the first left behind; in its own interpreter it never does.
LEAKED: list[int] = []


def record_pid(n) -> None:
    """Record the process the job ran in."""
    Metric("pid", dtype=int).push(0, os.getpid())


def leak(n) -> None:
    """Record how much state was already there when this job started."""
    LEAKED.append(n)
    Metric("seen", dtype=int).push(0, len(LEAKED))


def explode(n) -> None:
    """Fail the way a user experiment would."""
    raise ValueError(f"n={n} is not supported")


def expire(n) -> None:
    """Leave without returning, as a job killed by the system would."""
    Metric("pid", dtype=int).push(0, os.getpid())
    os._exit(17)
