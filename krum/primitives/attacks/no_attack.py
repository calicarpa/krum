r"""No-op attack — honest baseline with no Byzantine gradients."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, stack

from . import Attack


class NoAttack(Attack):
    """Attack stub that always returns an empty Byzantine-gradient tensor.

    A no-op attack is required by the simulation loop (which expects an
    :class:`~krum.primitives.attacks.Attack` instance even when
    :math:`f = 0`). It produces zero malicious gradients regardless of
    the honest gradients, so the aggregator only ever sees honest
    inputs.
    """

    @classmethod
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        **specialized: Any,
    ) -> Tensor:
        """Generate Byzantine gradients.

        Args:
            honest_gradients: Sequence of :math:`h` gradient vectors, one per honest
                worker, each of shape :math:`(d,)`. Only the second dimension is used.
            out: Optional pre-allocated tensor. Ignored.
            f: Number of Byzantine gradients requested by the caller. Ignored.
            **specialized: Additional keyword arguments.

        Returns:
            Empty tensor of shape ``(0, d)`` on the same device and dtype
            as ``honest_gradients``.
        """
        stacked = stack(list(honest_gradients))
        return stacked.new_empty((0, stacked.shape[1]))
