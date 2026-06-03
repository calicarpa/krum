"""Sign-flip gradient attack.

Reference:
    Blanchard, Peva, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
    Stainer. "Machine learning with adversaries: Byzantine tolerant
    gradient descent." In Advances in Neural Information Processing
    Systems 30 (NIPS 2017).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, is_floating_point, stack

from . import Attack


class SignFlipAttack(Attack):
    """Sign-flip attack.

    Generates Byzantine gradients from the negative honest mean, optionally
    scaled by a positive factor. Intuitively, every Byzantine worker tries to
    make the aggregated gradient point in the opposite direction of the honest
    update.

    Args:
        honest_gradients: Sequence of 1-D tensors, one per honest worker.
        f: Number of Byzantine gradients to generate.
        scale: Non-negative scale applied to the sign-flipped honest mean.
            ``scale = 1`` sends the exact negative honest mean; larger values
            amplify the attack.

    Returns:
        Byzantine gradients of shape ``(f, d)``.

    Raises:
        ValueError: If ``scale`` or ``f`` is negative, or there are no honest
            gradients to average.
        TypeError: If the honest gradients do not use a floating-point dtype.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        *,
        f: int,
        scale: float = 1.0,
        **specialized: Any,
    ) -> Tensor:
        """Generate sign-flipped Byzantine gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            f: Number of Byzantine gradients to generate.
            scale: Non-negative scale applied to the sign-flipped honest mean.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape ``(f, d)``. The same sign-flipped honest
            mean is repeated ``f`` times.

        Raises:
            ValueError: If ``scale`` or ``f`` is negative, or there are no honest
                gradients to average.
            TypeError: If the honest gradients do not use a floating-point dtype.
        """
        if scale < 0:
            msg = f"Invalid sign-flip scale, got {scale!r}, expected scale >= 0"
            raise ValueError(msg)
        if f < 0:
            msg = f"Invalid number of Byzantine gradients to generate, got {f!r}, expected 0 <= f"
            raise ValueError(msg)
        if len(honest_gradients) == 0:
            msg = "Expected at least one honest gradient to compute the honest mean"
            raise ValueError(msg)
        stacked = stack(list(honest_gradients))
        if not is_floating_point(stacked):
            raise TypeError("Expected honest gradients to use a floating-point dtype")

        malicious_gradient = -scale * stacked.mean(dim=0)

        return malicious_gradient.repeat(f, 1)
