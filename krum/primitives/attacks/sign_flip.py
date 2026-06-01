"""Sign-flip gradient attack.

Reference:
    Blanchard, Peva, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
    Stainer. "Machine learning with adversaries: Byzantine tolerant
    gradient descent." In Advances in Neural Information Processing
    Systems 30 (NIPS 2017).
"""

from collections.abc import Sequence

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

    scale: float

    __slots__ = ("scale",)

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
        honest_gradients: Sequence[torch.Tensor],
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate sign-flipped Byzantine gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape ``(num_byzantine, d)``. The same
            sign-flipped honest mean is repeated ``num_byzantine`` times.

        Raises:
            ValueError: If ``num_byzantine`` is negative or there are no honest
                gradients to average.
            TypeError: If the honest gradients do not use a floating-point dtype.
        """
        if num_byzantine < 0:
            msg = (
                f"Invalid number of Byzantine gradients to generate, got {num_byzantine!r}, expected 0 <= num_byzantine"
            )
            raise ValueError(msg)
        if len(honest_gradients) == 0:
            msg = "Expected at least one honest gradient to compute the honest mean"
            raise ValueError(msg)
        stacked = torch.stack(list(honest_gradients))
        if not torch.is_floating_point(stacked):
            raise TypeError("Expected honest gradients to use a floating-point dtype")

        malicious_gradient = -self.scale * stacked.mean(dim=0)

        return malicious_gradient.repeat(num_byzantine, 1)
