"""Sign-flip attack."""

import torch

from .attack import Attack


class SignFlipAttack(Attack):
    """Sign-flip attack.

    Generates Byzantine gradients from the negative honest mean, optionally
    scaled by a positive factor.

    Args:
        scale: Scale applied to the sign-flipped honest mean.
    """

    def __init__(self, *, scale: float = 1.0) -> None:
        """Initialize the attack.

        Args:
            scale: Scale applied to the sign-flipped honest mean.
        """
        if scale < 0:
            msg = f"Invalid sign-flip scale, got {scale!r}, expected scale >= 0"
            raise ValueError(msg)
        self.scale = scale

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate sign-flipped Byzantine gradients.

        Args:
            honest_gradients: Tensor of shape (h, d) containing gradients from honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape (num_byzantine, d).
        """
        if honest_gradients.ndim != 2:
            raise ValueError("Expected a 2D tensor of honest gradients")
        if not torch.is_floating_point(honest_gradients):
            raise TypeError("Expected honest gradients to use a floating-point dtype")
        if num_byzantine < 0:
            msg = f"Invalid number of Byzantine gradients to generate, got {num_byzantine!r}, expected 0 <= num_byzantine"
            raise ValueError(msg)
        if honest_gradients.shape[0] == 0:
            msg = "Expected at least one honest gradient to compute the honest mean"
            raise ValueError(msg)

        malicious_gradient = -self.scale * honest_gradients.mean(dim=0)

        return malicious_gradient.repeat(num_byzantine, 1)
