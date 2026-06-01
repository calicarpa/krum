"""Gradient attacks that simulate Byzantine workers.

An :class:`~krum.primitives.attacks.attack.Attack` observes the gradients of
honest workers and produces gradients that mimic what adversarial (Byzantine)
workers would send. They are used to stress-test
:mod:`~krum.primitives.aggregators` rules.

Provided attacks:

* :class:`~krum.primitives.attacks.sign_flip.SignFlipAttack` — sends the
  sign-flipped honest mean (Blanchard et al., NIPS 2017).
* :class:`~krum.primitives.attacks.alie.ALIEAttack` — sends a mean-shifted
  gradient computed from exact honest coordinate-wise statistics
  (Baruch et al., ICML 2019).
"""

from krum.primitives.attacks.alie import ALIEAttack
from krum.primitives.attacks.attack import Attack
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.primitives.attacks.types import Direction

__all__ = ["ALIEAttack", "Attack", "Direction", "SignFlipAttack"]
