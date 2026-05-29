"""Tests for the Krum aggregator."""

import unittest

import torch

from krum.primitives.aggregators import Krum


class KrumTest(unittest.TestCase):
    """Test Krum aggregator."""

    def test_aggregate_selects_best_gradient(self) -> None:
        """Krum selects the gradient with the smallest sum of distances to its n-f-2 closest neighbors."""
        agg = Krum(n=5, f=1)
        grads = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [100.0, 100.0],
        ])
        result = agg.aggregate(grads)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(torch.equal(result, grads[1]))

    def test_aggregate_identical_gradients(self) -> None:
        """Krum returns the first gradient when all are identical."""
        agg = Krum(n=3, f=0)
        grads = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads[0]))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = Krum(n=5, f=1)
        grads = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]],
            dtype=torch.float64,
        )
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            Krum(n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            Krum(n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            Krum(n=5, f=10)

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 3."""
        with self.assertRaises(ValueError):
            Krum(n=3, f=1)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            Krum(5, 1)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when gradients.shape[0] != n."""
        agg = Krum(n=5, f=1)
        with self.assertRaises(ValueError):
            agg.aggregate(torch.tensor([[1.0], [2.0], [3.0]]))


if __name__ == "__main__":
    unittest.main()
