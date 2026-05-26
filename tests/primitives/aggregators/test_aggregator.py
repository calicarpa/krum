"""Tests for the base Aggregator class."""

import unittest

import torch

from krum.primitives.aggregators import Aggregator


class _ConcreteAggregator(Aggregator):
    """Minimal concrete aggregator for testing the base class."""

    def __init__(self, *, n, f):
        super().__init__(n=n, f=f)
        self.check()

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        return gradients.mean(0)


class AggregatorTest(unittest.TestCase):
    """Test Aggregator base class."""

    def test_aggregate_through_call(self) -> None:
        """__call__ delegates to aggregate."""
        agg = _ConcreteAggregator(n=3, f=1)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = agg(grads)
        expected = grads.mean(0)
        self.assertTrue(torch.equal(result, expected))

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            _ConcreteAggregator(n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            _ConcreteAggregator(n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            _ConcreteAggregator(n=5, f=10)

    def test_upper_bound_returns_nan(self) -> None:
        """upper_bound returns NaN by default."""
        agg = _ConcreteAggregator(n=5, f=1)
        self.assertTrue(torch.isnan(torch.tensor(agg.upper_bound())))

    def test_influence_ratio_returns_nan(self) -> None:
        """influence_ratio returns NaN by default."""
        agg = _ConcreteAggregator(n=5, f=1)
        honest = torch.ones((3, 4))
        byzantine = torch.zeros((2, 4))
        result = agg.influence_ratio(honest, byzantine)
        self.assertTrue(torch.isnan(torch.tensor(result)))

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            _ConcreteAggregator(3, 1)


if __name__ == "__main__":
    unittest.main()
