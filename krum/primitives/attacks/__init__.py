"""Gradient attacks that simulate Byzantine workers.

An :class:`Attack` observes the gradients of honest workers and produces
gradients that mimic what adversarial (Byzantine) workers would send. They are
used to stress-test :mod:`~krum.primitives.aggregators` rules.

Concrete attacks live in their own submodules and are imported directly from
them (e.g. ``from krum.primitives.attacks.sign_flip import SignFlipAttack``).
The package root only exposes the :class:`Attack` base class, imported by
submodules with ``from . import Attack``.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class Attack(ABC):
    """Base class for gradient attacks in Byzantine-resilient distributed learning.

    An attack observes the gradients produced by honest workers and returns
    gradients that a Byzantine worker (or workers) would send to the
    aggregator. Subclasses are invoked as ``attack(honest_gradients,
    num_byzantine)`` and must implement :meth:`generate`.
    """

    # No instance state of its own; the empty slots let subclasses' __slots__
    # take effect (a non-slotted base would still give instances a __dict__).
    __slots__ = ()

    @abstractmethod
    def generate(
        self,
        honest_gradients: Sequence[torch.Tensor],
        num_byzantine: int,
    ) -> torch.Tensor:
        """Generate Byzantine gradients from observed honest gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape ``(num_byzantine, d)``.
        """
        pass

    def __call__(
        self,
        honest_gradients: Sequence[torch.Tensor],
        num_byzantine: int,
    ) -> torch.Tensor:
        """Call :meth:`generate` to produce Byzantine gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            num_byzantine: Number of Byzantine gradients to generate.

        Returns:
            Byzantine gradients of shape ``(num_byzantine, d)``.
        """
        return self.generate(honest_gradients, num_byzantine)
