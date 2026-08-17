"""Krum — Byzantine-resilient aggregation rules for distributed machine learning.

The package is organized in two layers:

* :mod:`krum.primitives.aggregators` exposes stateless, classmethod-based
  Gradient Aggregation Rules (GARs).
* :mod:`krum.primitives.attacks` exposes gradient attacks that simulate Byzantine
  workers when evaluating the robustness of an aggregator.
* :mod:`krum.primitives.models` provides a :class:`~krum.primitives.models.Model`
  wrapper that exposes a ``torch.nn.Module``'s parameters and gradients as
  zero-copy flat tensors, which is the data layout the aggregators consume.

The :mod:`krum.simulations` package ships the original training simulations
from the papers that introduced each rule.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("krum")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Running from source without installation
