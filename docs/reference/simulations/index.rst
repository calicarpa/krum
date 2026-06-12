Simulations
===========

The :mod:`krum.simulations.centralised` package provides a parameter-server
distributed SGD simulation framework for reproducing published Byzantine-resilient
learning experiments.

Architecture
------------

:class:`~krum.simulations.centralised.CentralisedSimulation` is the base class
implementing the full training lifecycle — model initialisation, IID data sharding,
synchronous rounds of gradient computation and aggregation, and configurable
evaluation. Protocol-specific behaviour is composed via the ``evaluate_fn``
callable and ``lr_schedule`` parameter, following composition over inheritance.

Two paper-specific subclasses bundle pre-configured
:ref:`evaluators and schedules <sim_protocols>`:

- :class:`~krum.simulations.centralised.HiddenVulnerabilitySimulation` —
  reproduces experiments from El Mhamdi et al. (ICML 2018).
- :class:`~krum.simulations.centralised.KrumSimulation` —
  reproduces experiments from Blanchard et al. (NIPS 2017).

.. _sim_protocols:

Available Simulations:
----------------------
   
.. toctree::
   :maxdepth: 1

   classes/centralised
   classes/hidden_vulnerability_icml_2018
   classes/krum_nips_2017
