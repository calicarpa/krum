"""Tests for the base Aggregator class."""

import unittest
from collections.abc import Sequence
from typing import Any

import torch

from krum.primitives.aggregators import Aggregator


class _ConcreteAggregator(Aggregator):
    """Minimal concrete aggregator for testing the base class."""

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[torch.Tensor] | torch.Tensor,
        /,
        out: torch.Tensor | None = None,
        **specialized: Any,
    ) -> torch.Tensor:
        return torch.stack(list(gradients)).mean(0)


class AggregatorTest(unittest.TestCase):
    """Test Aggregator base class."""

    def test_aggregate(self) -> None:
        """Aggregate works as expected."""
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _ConcreteAggregator.aggregate(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
