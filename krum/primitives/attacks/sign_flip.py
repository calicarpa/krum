"""Sign-flip gradient attack.

Reference:
    Blanchard, Peva, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
    Stainer. "Machine learning with adversaries: Byzantine tolerant
    gradient descent." In Advances in Neural Information Processing
    Systems 30 (NIPS 2017).
"""

import torch

from . import Attack


class SignFlipAttack(Attack):
    """Sign-flip attack.

    Generates Byzantine gradients from the negative honest mean, optionally
    scaled by a positive factor. Intuitively, every Byzantine worker tries
    to make the aggregated gradient point in the opposite direction of the
    honest update.

    Args:
        scale: Non-negative scale applied to the sign-flipped honest mean.
            ``scale = 1`` sends the exact negative honest mean; larger
            values amplify the attack.

    Raises:
        ValueError: If ``scale`` is negative.
    """

    def __init__(self, *, scale: float = 1.0) -> None:
        """Initialize the attack.

        Args:
            scale: Non-negative scale applied to the sign-flipped honest
                mean.
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
            honest_gradients: Tensor of shape ``(h, d)`` containing gradients
                from the ``h`` honest workers.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape ``(num_byzantine, d)``. The same
            sign-flipped honest mean is repeated ``num_byzantine`` times.

        Raises:
            ValueError: If ``honest_gradients`` is not 2-D, ``num_byzantine``
                is negative, or there are no honest gradients to average.
            TypeError: If ``honest_gradients`` does not use a floating-point
                dtype.
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
        if honest_gradients.shape[0] == 0:
            msg = "Expected at least one honest gradient to compute the honest mean"
            raise ValueError(msg)

        malicious_gradient = -self.scale * honest_gradients.mean(dim=0)

        return malicious_gradient.repeat(num_byzantine, 1)
