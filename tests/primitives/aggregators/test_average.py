"""Tests for the Average aggregator."""

import unittest

import torch

from krum.primitives.aggregators.average import Average


class AverageTest(unittest.TestCase):
    """Test Average aggregator."""

    def test_aggregate_computes_mean(self) -> None:
        """Aggregate returns the coordinate-wise mean."""
        agg = Average()
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = agg.aggregate(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with a single gradient returns it unchanged."""
        agg = Average()
        grads = torch.tensor([[7.0, 8.0, 9.0]])
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads.squeeze(0)))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = Average()
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
