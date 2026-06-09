"""Byzantine-resilient gradient aggregation rules.

In distributed learning with ``n`` workers of which up to ``f`` may be
Byzantine (adversarial or faulty), the aggregator is the single point that
determines whether the training converges. Each rule in this module accepts
one gradient per worker and produces a single aggregated gradient that is
robust to the ``f`` worst outliers.

All aggregators are **stateless**: each rule is a ``@classmethod`` invoked
directly on the class. Specialized parameters (``f``, ``n``, ``m``) are
keyword-only.

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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from torch import Tensor


class Aggregator(ABC):
    """Abstract base class for stateless gradient aggregation rules.

    Subclasses implement :meth:`aggregate` as a ``@classmethod`` — no instance
    state is required, and the caller invokes the rule directly on the class.
    The first positional argument is the worker gradients; rule-specific
    hyperparameters (``f``, ``n``, ``m``) are keyword-only.
    """

    @classmethod
    @abstractmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        **specialized: Any,
    ) -> Tensor:
        """Aggregate the gradients into a single tensor.

        Args:
            gradients: Sequence of 1-D tensors containing one gradient per
                worker. Tensors are expected to share dtype and device.
            out: Optional pre-allocated tensor to write the result into.
                Must have shape ``(d,)`` and the correct ``dtype`` and
                ``device``. Passed through to the underlying PyTorch
                operation when supported.
            **specialized: Keyword-only arguments specific to each
                aggregation rule (e.g. ``f``, ``n``, ``m``).

        Returns:
            Aggregated gradient of shape ``(d,)``.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
