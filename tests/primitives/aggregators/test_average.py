"""Tests for the Average aggregator."""

import unittest

import torch

from krum.primitives.aggregators import Average


class AverageTest(unittest.TestCase):
    """Test Average aggregator."""

    def test_aggregate_computes_mean(self) -> None:
        """Aggregate returns the coordinate-wise mean."""
        agg = Average(n=3, f=1)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = agg.aggregate(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with a single gradient returns it unchanged."""
        agg = Average(n=1, f=0)
        grads = torch.tensor([[7.0, 8.0, 9.0]])
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads.squeeze(0)))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = Average(n=3, f=1)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_influence_ratio_all_honest(self) -> None:
        """influence_ratio is zero when there are no Byzantine gradients."""
        agg = Average(n=3, f=0)
        honest = torch.ones((3, 4))
        byzantine = torch.empty((0, 4))
        self.assertEqual(agg.influence_ratio(honest, byzantine), 0.0)

    def test_influence_ratio_all_byzantine(self) -> None:
        """influence_ratio is 1 when all gradients are Byzantine."""
        agg = Average(n=3, f=3)
        honest = torch.empty((0, 4))
        byzantine = torch.ones((3, 4))
        self.assertEqual(agg.influence_ratio(honest, byzantine), 1.0)

    def test_upper_bound_is_nan(self) -> None:
        """upper_bound returns NaN for Average."""
        agg = Average(n=5, f=1)
        self.assertTrue(torch.isnan(torch.tensor(agg.upper_bound())))

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            Average(3, 1)


if __name__ == "__main__":
    unittest.main()
