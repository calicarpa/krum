"""Sign-flip gradient attack.

Reference:
    Cong Xie, Sanmi Koyejo, and Indranil Gupta.
    "Fall of Empires: Breaking Byzantine-tolerant SGD by Tailored Attacks."
    In Proceedings of the 22nd International Conference on Artificial Intelligence
    and Statistics (AISTATS 2019).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, is_floating_point, mul, stack

from . import Attack


class SignFlipAttack(Attack):
    """Sign-flip attack using the negated honest mean.

    Generates Byzantine gradients by flipping the sign of the coordinate-wise
    honest mean and optionally scaling the result. Every Byzantine worker
    sends the same vector, pointing in the opposite direction of the honest
    update, which makes this one of the simplest yet effective baselines for
    evaluating Byzantine-resilient aggregators.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        scale: float = 1.0,
        **specialized: Any,
    ) -> Tensor:
        """Generate Byzantine gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            out: Optional pre-allocated tensor of shape ``(f, d)`` to write the
                result into and return.
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

        # Tile the sign-flipped mean to (f, d) through ``mul``'s ``out=`` so the
        # buffer-reuse and wrong-shape (resize) behavior matches the aggregators.
        honest_mean = stacked.mean(dim=0)
        return mul(honest_mean.unsqueeze(0).expand(f, -1), -scale, out=out)
