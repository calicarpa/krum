"""No-op attack — honest baseline with no Byzantine gradients.

A :class:`NoAttack` is the canonical placeholder for the "no adversary"
configuration of a distributed-SGD run: it returns an empty tensor for
every call to :meth:`generate`, so the simulation includes no Byzantine
gradients in the aggregator input. It is used as the honest baseline in
the experiments of El Mhamdi et al. (ICML 2018).
"""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, stack

from . import Attack


class NoAttack(Attack):
    """Attack stub that always returns an empty Byzantine-gradient tensor.

    A no-op attack is required by the simulation loop (which expects an
    :class:`~krum.primitives.attacks.attack.Attack` instance even when
    :math:`f = 0`). It produces zero malicious gradients regardless of
    the honest gradients, so the aggregator only ever sees honest
    inputs.

    This corresponds to the "Average baseline" runs of El Mhamdi et al.
    (ICML 2018), where :math:`f = 0` and no defense is applied.

    Args:
        honest_gradients: Sequence of 1-D tensors, one per honest worker.
        f: Number of Byzantine gradients to generate. Ignored.

    Returns:
        Empty tensor of shape ``(0, d)``.
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
        """Return an empty Byzantine-gradient tensor of shape ``(0, d)``.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``. Only the second dimension is used.
            out: Optional pre-allocated tensor. Ignored.
            f: Number of Byzantine gradients requested by the caller. Ignored.
            **specialized: Additional keyword arguments.

        Returns:
            Empty tensor of shape ``(0, d)`` on the same device and dtype
            as ``honest_gradients``.
        """
        stacked = stack(list(honest_gradients))
        return stacked.new_empty((0, stacked.shape[1]))
