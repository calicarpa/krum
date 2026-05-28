"""Configuration for the MoNNA simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MonnaConfig:
    """Parameters controlling one MoNNA simulation.

    Args:
        num_honest: Number of honest workers.
        num_byzantine: Number of Byzantine workers to inject.
        learning_rate: Local update step size.
        beta: Polyak momentum coefficient from the paper.
    """

    num_honest: int
    num_byzantine: int
    learning_rate: float
    beta: float = 0.99

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_honest < 1:
            raise ValueError(f"Expected at least one honest worker, got {self.num_honest!r}")
        if self.num_byzantine < 0:
            raise ValueError(f"Expected non-negative Byzantine worker count, got {self.num_byzantine!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"Expected positive learning rate, got {self.learning_rate!r}")
        if self.beta < 0 or self.beta >= 1:
            raise ValueError(f"Expected beta in [0, 1), got {self.beta!r}")
