"""Tests for the Median aggregator."""

import unittest

import torch

from krum.primitives.aggregators.median import Median


class MedianTest(unittest.TestCase):
    """Test Median aggregator."""

    def test_aggregate_computes_coordinate_wise_median(self) -> None:
        """Aggregate returns the coordinate-wise median."""
        agg = Median()
        grads = [torch.tensor([1.0, 9.0, 2.0]), torch.tensor([3.0, 1.0, 5.0]), torch.tensor([5.0, 4.0, 8.0])]
        result = agg.aggregate(grads)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_odd_number_of_gradients(self) -> None:
        """Aggregate handles an odd number of gradients."""
        agg = Median()
        grads = [
            torch.tensor([1.0]),
            torch.tensor([3.0]),
            torch.tensor([5.0]),
            torch.tensor([2.0]),
            torch.tensor([4.0]),
        ]
        result = agg.aggregate(grads)
        self.assertEqual(result.item(), 3.0)

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with one gradient returns it."""
        agg = Median()
        grads = [torch.tensor([7.0, 8.0])]
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads[0]))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = Median()
        grads = [
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            torch.tensor([5.0, 6.0], dtype=torch.float64),
        ]
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
