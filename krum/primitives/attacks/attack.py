"""Base class for gradient attacks."""

from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    """Base class for gradient attacks in Byzantine-resilient distributed learning.

    An attack observes the gradients produced by honest workers and returns
    gradients that a Byzantine worker (or workers) would send to the
    aggregator. Subclasses are invoked as ``attack(honest_gradients,
    num_byzantine)`` and must implement :meth:`generate`.
    """

    @abstractmethod
    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients from observed honest gradients.

        Args:
            honest_gradients: Tensor of shape ``(h, d)`` containing gradients
                from the ``h`` honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape ``(num_byzantine, d)``.
        """
        pass

    def __call__(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Call :meth:`generate` to produce Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape ``(h, d)`` containing gradients
                from the ``h`` honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape ``(num_byzantine, d)``.
        """
        return self.generate(honest_gradients, num_byzantine)
