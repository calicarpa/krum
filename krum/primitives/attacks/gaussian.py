r"""Gaussian noise gradient attack."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, randn, stack

from . import Attack


class GaussianAttack(Attack):
    """Gaussian Byzantine attack.

    Byzantine workers send vectors drawn from an isotropic Gaussian
    distribution with configurable mean and standard deviation.
    This attack is independent of the honest gradients.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        mu: float = 0.0,
        std: float = 200.0,
        **specialized: Any,
    ) -> Tensor:
        r"""Generate Gaussian Byzantine gradients.

        Args:
            honest_gradients: Sequence of :math:`h` gradient vectors, one per honest
                worker, each of shape :math:`(d,)`. Only the second dimension is used.
            out: Optional pre-allocated tensor of shape :math:`(f, d)` to write the
                result into and return.
            f: Number of Byzantine gradients to generate.
            mu: Mean of the Gaussian noise.
            std: Standard deviation of the Gaussian noise.
            **specialized: Additional keyword arguments.

        Returns:
            Byzantine gradients of shape ``(f, d)``. When ``f = 0``, returns an
            empty tensor of shape ``(0, d)``.

        Raises:
            ValueError: If :math:`\sigma` or :math:`f` is negative, or there are no honest
                gradients.
            TypeError: If the honest gradients do not use a floating-point dtype.
        """
        if std < 0:
            msg = f"Invalid standard deviation, got {std!r}, expected std >= 0"
            raise ValueError(msg)
        if f < 0:
            msg = f"Invalid number of Byzantine gradients to generate, got {f!r}, expected 0 <= f"
            raise ValueError(msg)
        if len(honest_gradients) == 0:
            msg = "Expected at least one honest gradient to determine output shape"
            raise ValueError(msg)

        stacked = stack(list(honest_gradients))

        if f == 0:
            return stacked.new_empty((0, stacked.shape[1]))

        d = stacked.shape[1]
        result = mu + randn(f, d, device=stacked.device, dtype=stacked.dtype) * std
        if out is not None:
            return out.copy_(result)
        return result
