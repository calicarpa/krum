"""Peer-to-peer decentralised learning simulation.

Each honest worker holds its own model. Each round runs two phases:

#. **Local optimisation** — each worker computes a gradient on its local batch
   and updates its own model.
#. **Model mixing** — each worker gathers models from other nodes and replaces
   its model with an aggregate of the received set.

Two abstract seams let protocols vary the local update rule (e.g. momentum-SGD)
and the communication topology (which models each worker receives).
"""

from krum.simulations.decentralised.base import DecentralisedSimulation, StepResult
from krum.simulations.decentralised.monna_icml_2023 import ByzantineReach, MonnaSimulation

__all__ = ["ByzantineReach", "DecentralisedSimulation", "MonnaSimulation", "StepResult"]
