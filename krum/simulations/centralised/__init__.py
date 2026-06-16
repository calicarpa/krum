"""Parameter-server distributed SGD simulation.

Each synchronous round follows the same pattern:

#. Honest workers compute a gradient on their local data shard.
#. Byzantine workers craft adversarial gradients.
#. The aggregator combines all :math:`n` gradients into a single update.
#. The aggregated update is applied via an SGD step.
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
