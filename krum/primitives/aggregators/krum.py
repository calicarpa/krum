"""Krum aggregation rule."""

from . import MultiKrum


class Krum(MultiKrum):
    """Krum aggregation rule.

    Args:
        n: Total number of workers.
        f: Number of Byzantine workers to tolerate.
    """

    def __init__(self, *, n: int, f: int) -> None:
        """Initialize Krum aggregator.

        Args:
            n: Total number of workers.
            f: Number of Byzantine workers to tolerate.
        """
        super().__init__(n=n, f=f, m=1)
