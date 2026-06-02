"""Byzantine-resilient gradient aggregation rules.

All aggregators are **stateless**: each rule is a classmethod that can be
called in three equivalent ways:

* ``Aggregate.aggregate(gradients, **kwargs)`` — explicit classmethod call
* ``Aggregate(gradients, **kwargs)`` — call the class directly
* ``Aggregate()(gradients, **kwargs)`` — call an instance

Specialized parameters (``f``, ``n``, ``m``) are keyword-only.

Available rules:

* :class:`~krum.primitives.aggregators.average.Average` — plain mean (no robustness, baseline).
* :class:`~krum.primitives.aggregators.median.Median` — coordinate-wise median.
* :class:`~krum.primitives.aggregators.trimmed_mean.TrimmedMean` — coordinate-wise
  trimmed mean.
* :class:`~krum.primitives.aggregators.krum.Krum` — single-gradient selection
  (Blanchard et al., NIPS 2017).
* :class:`~krum.primitives.aggregators.multikrum.MultiKrum` — multi-gradient
  averaging variant of Krum (Blanchard et al., NIPS 2017).
* :class:`~krum.primitives.aggregators.bulyan.Bulyan` — two-stage Krum +
  trimmed mean (Mhamdi et al., ICML 2018).
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch


class Aggregator(ABC):
    """Abstract base class for stateless gradient aggregation rules.

    Subclasses implement :meth:`aggregate` as a ``@classmethod`` — no instance
    state is required.  The aggregator can be called in three equivalent ways:

    * ``Average.aggregate(gradients, **kwargs)`` — explicit classmethod call
    * ``Average(gradients, **kwargs)`` — call the class directly
    * ``Average()(gradients, **kwargs)`` — call an instance
    """

    @classmethod
    @abstractmethod
    def aggregate(cls, gradients: Sequence[torch.Tensor], /, **specialized: Any) -> torch.Tensor:
        """Aggregate the gradients into a single tensor.

        Args:
            gradients: Sequence of 1-D tensors containing one gradient per
                worker. Tensors are expected to share dtype and device.
            **specialized: Keyword-only arguments specific to each
                aggregation rule (e.g. ``f``, ``n``, ``m``).

        Returns:
            Aggregated gradient of shape ``(d,)``.
        """
        pass
