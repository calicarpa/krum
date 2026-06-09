"""Byzantine-resilient gradient aggregation rules.

In distributed learning with :math:`n` workers of which up to :math:`f` may be
Byzantine (adversarial or faulty), the aggregator is the single point that
determines whether the training converges. Each rule in this module accepts
one gradient per worker and produces a single aggregated gradient that is
robust to the :math:`f` worst outliers.

All aggregators are **stateless**: each rule is a ``@classmethod`` invoked
directly on the class. Specialized parameters (:math:`f`, :math:`n`, :math:`m`) are
keyword-only.
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
    hyperparameters (:math:`f`, :math:`n`, :math:`m`) are keyword-only.
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
                Must have shape :math:`(d,)` and the correct ``dtype`` and
                ``device``. Passed through to the underlying PyTorch
                operation when supported.
            **specialized: Keyword-only arguments specific to each
                aggregation rule (e.g. :math:`f`, :math:`n`, :math:`m`).

        Returns:
            Aggregated gradient of shape :math:`(d,)`.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
