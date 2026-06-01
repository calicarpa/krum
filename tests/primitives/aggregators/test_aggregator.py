"""Tests for the base Aggregator class."""

import unittest
from collections.abc import Sequence

import torch

from krum.primitives.aggregators import Aggregator


class _ConcreteAggregator(Aggregator):
    """Minimal concrete aggregator for testing the base class."""

    def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(gradients)).mean(0)


class AggregatorTest(unittest.TestCase):
    """Test Aggregator base class."""

    def test_aggregate_through_call(self) -> None:
        """__call__ delegates to aggregate."""
        agg = _ConcreteAggregator()
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = agg(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
