"""No-op attack — honest baseline with no Byzantine gradients.

A :class:`NoAttack` is the canonical placeholder for the "no adversary"
configuration of a distributed-SGD run: it returns an empty tensor for
every call to :meth:`generate`, so the simulation includes no Byzantine
gradients in the aggregator input. It is used as the honest baseline in
the experiments of El Mhamdi et al. (ICML 2018).
"""

from __future__ import annotations

import torch

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
    """

    def generate(
        self,
        honest_gradients: torch.Tensor,
        num_byzantine: int,
    ) -> torch.Tensor:
        """Return an empty Byzantine-gradient tensor of shape ``(0, d)``.

        Args:
            honest_gradients: Honest gradient stack of shape ``(h, d)``.
                Only the second dimension is used.
            num_byzantine: Number of Byzantine gradients requested by the
                caller. Ignored — the method always returns an empty
                stack.

        Returns:
            Empty tensor of shape ``(0, d)`` on the same device and dtype
            as ``honest_gradients``.

        Raises:
            ValueError: If ``honest_gradients`` is not 2-D.
        """
        if honest_gradients.ndim != 2:
            raise ValueError("Expected a 2D tensor of honest gradients")
        return honest_gradients.new_empty((0, honest_gradients.shape[1]))
