"""Gaussian attack."""

import torch

from . import Attack


class GaussianAttack(Attack):
    """Gaussian Byzantine attack.

    Byzantine workers send vectors drawn from an isotropic Gaussian
    distribution with mean zero and configurable standard deviation.
    This attack is independent of the honest gradients.

    Args:
        std: Standard deviation of the Gaussian noise. Default 200
            as used in the Krum NIPS-2017 paper.
    """

    def __init__(self, *, std: float = 200.0) -> None:
        """Initialize the Gaussian attack.

        Args:
            std: Standard deviation of the Gaussian noise.
        """
        if std < 0:
            msg = f"Invalid standard deviation, got {std!r}, expected std >= 0"
            raise ValueError(msg)
        self.std = std

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Gaussian Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
                Not used by this attack, only needed for the output shape.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d).
        """
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

        d = honest_gradients.shape[1]
        return torch.randn(num_byzantine, d, device=honest_gradients.device, dtype=honest_gradients.dtype) * self.std
