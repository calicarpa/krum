r"""Full-gradient negation attack."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, stack

from . import Attack


class FullGradientNegationAttack(Attack):
    r"""Full-gradient negation Byzantine attack.

    Each Byzantine worker sends the negation of the full-dataset gradient
    scaled by a factor :math:`\kappa`. The attack assumes knowledge of the
    true gradient over the entire dataset.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        full_gradient: Tensor,
        kappa: float = 100.0,
        **specialized: Any,
    ) -> Tensor:
        r"""Generate Byzantine gradients.

        Args:
            honest_gradients: Sequence of :math:`h` gradient vectors, one per honest
                worker, each of shape :math:`(d,)`.
            out: Optional pre-allocated tensor of shape :math:`(f, d)` to write the
                result into and return.
            f: Number of Byzantine gradients to generate.
            full_gradient: Tensor of shape :math:`(d,)` containing the gradient
                computed over the entire dataset.
            kappa: Scale factor applied to the negated full gradient.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape :math:`(f, d)`. When :math:`f = 0`, returns an
            empty tensor of shape :math:`(0, d)`.

        Raises:
            ValueError: If :math:`\kappa` or :math:`f` is negative, there are no honest
                gradients, or ``full_gradient`` is not 1-D.
            TypeError: If the honest gradients do not use a floating-point dtype.
        """
        if kappa < 0:
            msg = f"Invalid kappa, got {kappa!r}, expected kappa >= 0"
            raise ValueError(msg)
        if f < 0:
            msg = f"Invalid number of Byzantine gradients to generate, got {f!r}, expected 0 <= f"
            raise ValueError(msg)
        if full_gradient.ndim != 1:
            raise ValueError("Expected a 1D tensor for the full gradient")
        if len(honest_gradients) == 0:
            msg = "Expected at least one honest gradient to determine output shape"
            raise ValueError(msg)

        stacked = stack(list(honest_gradients))

        if f == 0:
            return stacked.new_empty((0, stacked.shape[1]))

        malicious_gradient = -kappa * full_gradient.to(device=stacked.device, dtype=stacked.dtype)
        result = malicious_gradient.repeat(f, 1)
        if out is not None:
            return out.copy_(result)
        return result
