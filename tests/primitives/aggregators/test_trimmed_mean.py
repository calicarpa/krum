"""Tests for the Trimmed Mean aggregator."""

import unittest

import torch

from krum.primitives.aggregators import TrimmedMean


class TrimmedMeanTest(unittest.TestCase):
    """Test Trimmed Mean aggregator."""

    def test_aggregate_computes_coordinate_wise_trimmed_mean(self) -> None:
        """Aggregate returns the coordinate-wise trimmed mean."""
        agg = TrimmedMean(n=5, f=1)
        grads = torch.tensor([[1.0, 10.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0], [100.0, 1.0]])
        result = agg.aggregate(grads)
        expected = torch.tensor([3.0, 5.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_odd_number_of_gradients(self) -> None:
        """Aggregate handles an odd number of gradients."""
        agg = TrimmedMean(n=7, f=2)
        grads = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
        result = agg.aggregate(grads)
        self.assertAlmostEqual(result.item(), 4.0)

    def test_aggregate_all_f_trimmed(self) -> None:
        """Aggregate trims f outliers from both ends."""
        agg = TrimmedMean(n=5, f=2)
        grads = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0]])
        result = agg.aggregate(grads)
        self.assertAlmostEqual(result.item(), 2.0)

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = TrimmedMean(n=5, f=1)
        grads = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            dtype=torch.float64,
        )
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            TrimmedMean(n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            TrimmedMean(n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            TrimmedMean(n=5, f=10)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            TrimmedMean(5, 1)


if __name__ == "__main__":
    unittest.main()
