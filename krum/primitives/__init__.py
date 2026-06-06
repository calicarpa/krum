"""Core abstractions shared by aggregators, attacks, and simulations.

The :class:`Model` wrapper encapsulates a ``torch.nn.Module`` and exposes
its parameters and gradients as zero-copy flat tensors, which is the layout
the aggregation and attack rules in :mod:`krum.primitives.aggregators` and
:mod:`krum.primitives.attacks` operate on.
"""

from .model import Model

__all__ = ["Model"]
