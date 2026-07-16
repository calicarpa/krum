"""Tests for the Median aggregator."""

import unittest

import torch

from krum.primitives.aggregators.median import Median


class MedianTest(unittest.TestCase):
    """Test Median aggregator."""

    def test_aggregate_computes_coordinate_wise_median(self) -> None:
        """Aggregate returns the coordinate-wise median."""
        grads = torch.tensor([[1.0, 9.0, 2.0], [3.0, 1.0, 5.0], [5.0, 4.0, 8.0]])
        result = Median.aggregate(grads)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_odd_number_of_gradients(self) -> None:
        """Aggregate handles an odd number of gradients."""
        grads = torch.tensor([[1.0], [3.0], [5.0], [2.0], [4.0]])
        result = Median.aggregate(grads)
        self.assertEqual(result.item(), 3.0)

    def test_aggregate_even_number_of_gradients_averages_middle_two(self) -> None:
        """For an even worker count, the median averages the two middle values.

        Regression guard for the paper's "usual (one-dimensional) median"
        (Yin et al., Definition 2.1), which torch.median does not implement:
        it returns the lower of the two middle values instead of their mean.
        """
        grads = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        result = Median.aggregate(grads)
        self.assertEqual(result.item(), 2.5)

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with one gradient returns it."""
        grads = torch.tensor([[7.0, 8.0]])
        result = Median.aggregate(grads)
        self.assertTrue(torch.equal(result, torch.tensor([7.0, 8.0])))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        result = Median.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
