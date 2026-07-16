"""Tests for the Krum aggregator."""

import unittest

import torch

from krum.primitives.aggregators.krum import Krum


class KrumTest(unittest.TestCase):
    """Test Krum aggregator."""

    def test_aggregate_selects_best_gradient(self) -> None:
        """Krum selects the gradient with the smallest sum of distances to its n-f-2 closest neighbors."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]])
        result = Krum.aggregate(grads, n=5, f=1)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(torch.equal(result, torch.tensor([0.0, 0.0])))

    def test_aggregate_identical_gradients(self) -> None:
        """Krum returns the first gradient when all are identical."""
        grads = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = Krum.aggregate(grads, n=3, f=0)
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 2.0])))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]], dtype=torch.float64)
        result = Krum.aggregate(grads, n=5, f=1)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            Krum.aggregate(torch.tensor([[1.0]]), n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            Krum.aggregate(torch.tensor([[1.0]]), n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            Krum.aggregate(torch.tensor([[1.0]]), n=5, f=10)

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 3."""
        with self.assertRaises(ValueError):
            Krum.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=3, f=1)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            Krum.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=5, f=1)


if __name__ == "__main__":
    unittest.main()
