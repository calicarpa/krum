"""Omniscient attack."""

import torch

from . import Attack


class OmniscientAttack(Attack):
    """Omniscient Byzantine attack.

    Each Byzantine worker computes the gradient on the full dataset,
    then sends the opposite vector scaled by a factor kappa.
    As used in the Krum NIPS-2017 paper, Section 5 (Cost of Resilience).

    The full-dataset gradient must be updated after each training round
    by setting the ``full_gradient`` attribute.

    Args:
        kappa: Scale factor applied to the negated full gradient.
            Should be large enough to dominate the aggregation.
    """

    def __init__(self, *, kappa: float = 100.0) -> None:
        """Initialize the omniscient attack.

        Args:
            kappa: Scale factor applied to the negated full gradient.
        """
        if kappa < 0:
            msg = f"Invalid kappa, got {kappa!r}, expected kappa >= 0"
            raise ValueError(msg)
        self.kappa = kappa
        self.full_gradient: torch.Tensor | None = None

    def set_full_gradient(self, full_gradient: torch.Tensor) -> None:
        """Set the full-dataset gradient for the current round.

        Args:
            full_gradient: Tensor of shape (d,) containing the gradient
                computed over the entire dataset.
        """
        if full_gradient.ndim != 1:
            raise ValueError("Expected a 1D tensor for the full gradient")
        self.full_gradient = full_gradient

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate omniscient Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
                Not used by this attack.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d).

        Raises:
            RuntimeError: If the full gradient has not been set.
        """
        if self.full_gradient is None:
            raise RuntimeError("Full gradient has not been set via set_full_gradient()")
        if honest_gradients.ndim != 2:
            raise ValueError("Expected a 2D tensor of honest gradients")
        if not torch.is_floating_point(honest_gradients):
            raise TypeError("Expected honest gradients to use a floating-point dtype")
        if num_byzantine < 0:
            msg = (
                f"Invalid number of Byzantine gradients to generate, got {num_byzantine!r}, expected 0 <= num_byzantine"
            )
            raise ValueError(msg)

        if num_byzantine == 0:
            return honest_gradients.new_empty((0, honest_gradients.shape[1]))

        malicious_gradient = -self.kappa * self.full_gradient.to(
            device=honest_gradients.device, dtype=honest_gradients.dtype
        )
        return malicious_gradient.repeat(num_byzantine, 1)
