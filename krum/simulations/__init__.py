"""Reusable simulations of Byzantine-resilient learning protocols.

Two architectures are supported: a single parameter-server that aggregates
:math:`n` worker gradients each round, and a peer-to-peer topology where each
honest worker mixes its model with those received from neighbors. Both share
the same primitives (aggregators, attacks, models) but differ in the training
loop structure and worker interaction pattern.
"""
