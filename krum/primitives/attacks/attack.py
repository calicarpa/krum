"""Base class for gradient attacks."""

from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    """Base class for gradient attacks in Byzantine-resilient distributed learning.

    An attack observes gradients from honest workers and generates gradients
    sent by Byzantine workers to an aggregator.
    """

    @abstractmethod
    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d).
        """
        pass

    def __call__(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d).
        """
        return self.generate(honest_gradients, num_byzantine)
