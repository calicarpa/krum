"""Core abstractions for Byzantine-resilient distributed learning.

Provides a zero-copy flat-tensor view of ``torch.nn.Module`` parameters and
gradients, stateless gradient aggregation rules, and Byzantine attack
strategies — the building blocks consumed by the simulation layer.
"""
