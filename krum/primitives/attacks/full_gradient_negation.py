r"""Full-gradient negation attack."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, stack

from . import Attack


class FullGradientNegationAttack(Attack):
    r"""Full-gradient negation attack.

    Each Byzantine worker sends the negation of the full-dataset gradient
    scaled by a factor :math:`\kappa`. The attack assumes knowledge of the
    true gradient over the entire dataset.

    .. note::
        No clipping or norm bound is applied to the malicious gradient:
        the attack returns :math:`-\kappa \cdot g_\text{full}` verbatim,
        so :math:`\kappa = 100` with a large :math:`d` can produce
        arbitrarily large-magnitude values (and ``NaN``/``Inf`` under
        :func:`torch.nn.functional.cross_entropy` with a Robbins-Monro
        learning-rate schedule). This is intentional — the point of the
        experiment is to stress-test the aggregator's robustness to
        unbounded adversaries. If you need a bounded variant, scale
        ``kappa`` down to your desired :math:`\ell_\infty` cap, or
        post-process the malicious gradient before injecting it.
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
            Byzantine gradients of shape ``(f, d)``. When ``f = 0``, returns an
            empty tensor of shape ``(0, d)``.

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
