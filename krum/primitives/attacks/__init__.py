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
from typing import Any

from torch import Tensor


class Attack(ABC):
    """Abstract base class for stateless gradient attacks.

    Subclasses implement :meth:`generate` as a ``@classmethod`` — no instance
    state is required, and the caller invokes the attack directly on the class.
    The first positional argument is the honest gradients; ``f`` (the number of
    Byzantine gradients to generate) and any attack-specific hyperparameters are
    keyword-only.
    """

    @classmethod
    @abstractmethod
    def generate(cls, honest_gradients: Sequence[Tensor] | Tensor, /, *, f: int, **specialized: Any) -> Tensor:
        """Generate Byzantine gradients from observed honest gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            f: Number of Byzantine gradients to generate.
            **specialized: Keyword-only arguments specific to each attack.

        Returns:
            Byzantine gradients of shape ``(f, d)``.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
