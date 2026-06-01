"""Tests for the base Aggregator class."""

import unittest

import torch

from krum.primitives.aggregators import Aggregator


class _ConcreteAggregator(Aggregator):
    """Minimal concrete aggregator for testing the base class."""

    @classmethod
    def aggregate(cls, gradients: list[torch.Tensor], /) -> torch.Tensor:
        return torch.stack(gradients).mean(0)


class AggregatorTest(unittest.TestCase):
    """Test Aggregator base class."""

    def test_aggregate(self) -> None:
        """Aggregate works as expected."""
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = _ConcreteAggregator.aggregate(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
