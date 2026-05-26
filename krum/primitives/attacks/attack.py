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

    def check(self, honest_gradients: torch.Tensor, num_byzantine: int) -> None:
        """Check input validity for attack rule.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Raises:
            ValueError: If inputs are invalid for the attack rule.
            TypeError: If honest gradients do not use a floating-point dtype.
        """
        if honest_gradients.ndim != 2:
            raise ValueError("Expected a 2D tensor of honest gradients")
        if not torch.is_floating_point(honest_gradients):
            raise TypeError("Expected honest gradients to use a floating-point dtype")
        if num_byzantine < 0:
            raise ValueError(
                f"Invalid number of Byzantine gradients to generate, got {num_byzantine!r}, expected 0 <= num_byzantine"
            )

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
