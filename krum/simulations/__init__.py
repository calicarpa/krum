"""Reusable simulations of Byzantine-resilient learning protocols.

The package is organised in two families:

* :mod:`krum.simulations.centralised` — parameter-server distributed SGD where
  a central aggregator combines all :math:`n` gradients into one update each
  round. Use this for classical Byzantine-robust aggregation experiments.

* :mod:`krum.simulations.decentralised` — peer-to-peer decentralised learning
  where each honest worker holds its own model and mixes it with models received
  from other workers. Use this for gossip-style protocol experiments.

Both families share the same primitive building blocks (aggregators, attacks,
models) but differ in the training loop's structure and the worker interaction
pattern.
"""
