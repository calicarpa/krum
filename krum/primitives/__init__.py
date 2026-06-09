"""Core abstractions shared by aggregators, attacks, and simulations.

This package provides the foundational building blocks for Byzantine-resilient
distributed learning experiments:

* :class:`Model` — zero-copy flat-tensor view of a ``torch.nn.Module``, the
  layout that aggregators and attacks operate on.
* :mod:`~krum.primitives.aggregators` — Byzantine-resilient gradient aggregation
  rules (Krum, MultiKrum, Bulyan, median, trimmed mean, …).
* :mod:`~krum.primitives.attacks` — gradient attacks that simulate Byzantine
  workers (sign-flip, ALIE, …).
"""

from .model import Model

__all__ = ["Model"]
