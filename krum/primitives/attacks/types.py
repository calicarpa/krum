"""Shared enum types for :mod:`krum.primitives.attacks`."""

from enum import Enum


class Direction(str, Enum):
    """Sign of the perturbation applied by an attack relative to the honest mean.

    Used by attacks that perturb a reference direction (e.g.
    :class:`~krum.primitives.attacks.alie.ALIEAttack`) to choose whether
    to shift the perturbation toward ``+infinity`` or ``-infinity``.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
