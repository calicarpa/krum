"""Base class shared by all simulation protocols."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StepResultT = TypeVar("StepResultT")


class Simulation(ABC, Generic[StepResultT]):
    """Abstract base for one-round-at-a-time learning simulations.

    A simulation advances through discrete rounds. Subclasses implement
    :meth:`step` to execute a single round and return a snapshot of it;
    :meth:`run` drives several rounds and collects the snapshots.

    The snapshot type is left to the concrete simulation via the generic
    parameter, e.g. ``class MonnaSimulation(Simulation[StepResult])``.
    """

    @abstractmethod
    def step(self) -> StepResultT:
        """Execute one simulation round and return its snapshot.

        Returns:
            A snapshot of the round, as defined by the concrete simulation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def run(self, rounds: int) -> list[StepResultT]:
        """Execute several simulation rounds.

        Args:
            rounds: Number of rounds to run; must be non-negative.

        Returns:
            One snapshot per round, in execution order.

        Raises:
            ValueError: If ``rounds`` is negative.
        """
        if rounds < 0:
            raise ValueError(f"Expected non-negative rounds, got {rounds!r}")
        return [self.step() for _ in range(rounds)]
