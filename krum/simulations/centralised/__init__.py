"""Centralised parameter-server simulation classes.

Provides the base :class:`CentralisedSimulation` and two paper-specific
subclasses: :class:`~krum.simulations.centralised.HiddenVulnerabilitySimulation`
(ICML 2018) and :class:`~krum.simulations.centralised.KrumSimulation`
(NIPS 2017).
"""

from krum.simulations.centralised.base import CentralisedSimulation as CentralisedSimulation
from krum.simulations.centralised.hidden_vulnerability_icml_2018 import (
    HiddenVulnerabilitySimulation as HiddenVulnerabilitySimulation,
)
from krum.simulations.centralised.krum_nips_2017 import KrumSimulation as KrumSimulation

__all__ = [
    "CentralisedSimulation",
    "HiddenVulnerabilitySimulation",
    "KrumSimulation",
]
