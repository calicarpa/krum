"""Gradient attacks."""

from krum.primitives.attacks.attack import Attack
from krum.primitives.attacks.inf import InfAttack
from krum.primitives.attacks.nan import NaNAttack

__all__ = ["Attack", "InfAttack", "NaNAttack"]
