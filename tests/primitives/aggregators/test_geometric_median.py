"""Tests for the GeometricMedian aggregator."""

import unittest

import torch

from krum.primitives.aggregators.geometric_median import GeometricMedian


class GeometricMedianTest(unittest.TestCase):
    """Test GeometricMedian aggregator."""

    def test_aggregate_centers_symmetric_square(self) -> None:
        """GeometricMedian returns the center of a symmetric square."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = GeometricMedian.aggregate(grads, n=4, f=0)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(torch.allclose(result, torch.tensor([0.5, 0.5]), atol=1e-4))

    def test_aggregate_minimizes_distance_sum(self) -> None:
        """GeometricMedian returns a point with no larger distance sum than any submitted vector."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
        result = GeometricMedian.aggregate(grads, n=4, f=1)
        result_sum = torch.cdist(result.unsqueeze(0), grads).sum().item()
        submitted_sums = torch.cdist(grads, grads).sum(dim=1)
        self.assertLessEqual(result_sum, submitted_sums.min().item() + 1e-3)

    def test_aggregate_resists_outlier(self) -> None:
        """GeometricMedian stays within the honest interval despite one extreme outlier."""
        grads = torch.tensor([[0.0], [0.5], [1.0], [100.0]])
        result = GeometricMedian.aggregate(grads, n=4, f=1)
        self.assertEqual(result.shape, (1,))
        self.assertGreaterEqual(result.item(), 0.4)
        self.assertLessEqual(result.item(), 1.1)

    def test_aggregate_single_gradient(self) -> None:
        """GeometricMedian with a single gradient returns it unchanged."""
        grads = torch.tensor([[7.0, 8.0, 9.0]])
        result = GeometricMedian.aggregate(grads, n=1, f=0)
        self.assertTrue(torch.allclose(result, grads[0]))

    def test_aggregate_writes_into_out(self) -> None:
        """GeometricMedian writes the result into the pre-allocated out buffer."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        out = torch.empty(2)
        result = GeometricMedian.aggregate(grads, n=4, f=0, out=out)
        self.assertIs(result, out)
        self.assertTrue(torch.allclose(out, torch.tensor([0.5, 0.5]), atol=1e-4))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        result = GeometricMedian.aggregate(grads, n=4, f=0)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            GeometricMedian.aggregate(torch.tensor([[1.0]]), n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            GeometricMedian.aggregate(torch.tensor([[1.0]]), n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            GeometricMedian.aggregate(torch.tensor([[1.0]]), n=5, f=10)

    def test_check_rejects_non_positive_nu(self) -> None:
        """Check raises ValueError when nu is not positive."""
        with self.assertRaises(ValueError):
            GeometricMedian.aggregate(torch.tensor([[1.0]]), n=1, f=0, nu=0.0)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            GeometricMedian.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=5, f=1)


if __name__ == "__main__":
    unittest.main()
