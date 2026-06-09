r"""Gradient attacks that simulate Byzantine (adversarial) workers.

An :class:`Attack` observes the gradients of honest workers and produces
gradients that mimic what adversarial (Byzantine) workers would send. They are
used to stress-test :mod:`~krum.primitives.aggregators` rules.

Available attacks:

* :class:`~krum.primitives.attacks.sign_flip.SignFlipAttack` — scaled opposite of the honest mean :math:`-\lambda \bar{g}`.
* :class:`~krum.primitives.attacks.alie.ALIEAttack` — mean-shifted using coordinate-wise statistics :math:`\bar{g} \pm z \cdot \sigma` (Baruch et al., NeurIPS 2019).
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
    def generate(
        cls,
        honest_gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        f: int,
        **specialized: Any,
    ) -> Tensor:
        r"""Generate Byzantine gradients from observed honest gradients.

        Args:
            honest_gradients: Sequence of ``h`` gradient vectors, one per honest
                worker, each of shape ``(d,)``.
            out: Optional pre-allocated tensor of shape ``(f, d)`` to write the
                result into and return, reusing its storage.
            f: Number of Byzantine gradients to generate.
            **specialized: Keyword-only arguments specific to each attack.

        Returns:
            Byzantine gradients of shape ``(f, d)``.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
