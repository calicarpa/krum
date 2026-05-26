"""Tests for the Median aggregator."""

import unittest

import torch

from krum.primitives.aggregators import Median


class MedianTest(unittest.TestCase):
    """Test Median aggregator."""

    def test_aggregate_computes_coordinate_wise_median(self) -> None:
        """Aggregate returns the coordinate-wise median."""
        agg = Median(n=3, f=1)
        grads = torch.tensor([[1.0, 9.0, 2.0], [3.0, 1.0, 5.0], [5.0, 4.0, 8.0]])
        result = agg.aggregate(grads)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_odd_number_of_gradients(self) -> None:
        """Aggregate handles an odd number of gradients."""
        agg = Median(n=5, f=2)
        grads = torch.tensor([[1.0], [3.0], [5.0], [2.0], [4.0]])
        result = agg.aggregate(grads)
        self.assertEqual(result.item(), 3.0)

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with one gradient returns it."""
        agg = Median(n=1, f=0)
        grads = torch.tensor([[7.0, 8.0]])
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads.squeeze(0)))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = Median(n=3, f=1)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_influence_ratio_no_byzantine_influence(self) -> None:
        """Influence_ratio is zero when Byzantine grads are far from the median."""
        agg = Median(n=5, f=2)
        honest = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        byzantine = torch.tensor([[100.0, 100.0], [200.0, 200.0]])
        result = agg.influence_ratio(honest, byzantine)
        self.assertEqual(result, 0.0)

    def test_influence_ratio_full_byzantine_influence(self) -> None:
        """Influence_ratio is 1 when Byzantine grads dominate the median."""
        agg = Median(n=5, f=3)
        honest = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        byzantine = torch.tensor([[1.5, 15.0], [1.5, 15.0], [1.5, 15.0]])
        result = agg.influence_ratio(honest, byzantine)
        self.assertEqual(result, 1.0)

    def test_upper_bound_returns_expected_value(self) -> None:
        """upper_bound returns 1 / sqrt(n - f)."""
        agg = Median(n=10, f=2)
        expected = 1.0 / (10.0 - 2.0) ** 0.5
        self.assertAlmostEqual(agg.upper_bound(), expected)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            Median(3, 1)


if __name__ == "__main__":
    unittest.main()
