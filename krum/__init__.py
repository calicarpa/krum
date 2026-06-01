"""Krum — Byzantine-resilient aggregation rules for distributed machine learning.

The package is organized in two layers:

* :mod:`krum.primitives.aggregators` exposes stateless, classmethod-based
  Gradient Aggregation Rules (GARs) such as :class:`~krum.primitives.aggregators.krum.Krum`,
  :class:`~krum.primitives.aggregators.multikrum.MultiKrum`,
  :class:`~krum.primitives.aggregators.bulyan.Bulyan`,
  :class:`~krum.primitives.aggregators.median.Median`,
  :class:`~krum.primitives.aggregators.trimmed_mean.TrimmedMean`, and
  :class:`~krum.primitives.aggregators.average.Average`.
* :mod:`krum.primitives.attacks` exposes gradient attacks (e.g.
  :class:`~krum.primitives.attacks.sign_flip.SignFlipAttack`,
  :class:`~krum.primitives.attacks.alie.ALIEAttack`) that simulate Byzantine
  workers when evaluating the robustness of an aggregator.
* :mod:`krum.primitives.model` provides a :class:`~krum.primitives.model.Model`
  wrapper that exposes a ``torch.nn.Module``'s parameters and gradients as
  zero-copy flat tensors, which is the data layout the aggregators consume.

The :mod:`krum.simulations` package ships the original training simulations
from the papers that introduced each rule.
"""
