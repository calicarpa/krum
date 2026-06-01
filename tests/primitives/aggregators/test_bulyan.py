"""Tests for the Bulyan aggregator."""

import unittest

import torch

from krum.primitives.aggregators.bulyan import Bulyan


class BulyanTest(unittest.TestCase):
    """Test Bulyan aggregator."""

    def test_aggregate_returns_gradient(self) -> None:
        """Bulyan returns a gradient of the expected shape."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        result = Bulyan.aggregate(grads, n=7, f=1, m=4)
        self.assertEqual(result.shape, (2,))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
            dtype=torch.float64,
        )
        result = Bulyan.aggregate(grads, n=7, f=1, m=4)
        self.assertEqual(result.dtype, torch.float64)

    def test_default_m(self) -> None:
        """M defaults to n - f - 2 when not provided."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        result = Bulyan.aggregate(grads, n=7, f=1)
        self.assertEqual(result.shape, (2,))

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=5, f=10)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            Bulyan(n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            Bulyan(n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            Bulyan(n=5, f=10)

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 4f + 3."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]), n=5, f=1)

    def test_aggregate_rejects_byzantine_outlier(self) -> None:
        """Bulyan produces a result close to the honest majority, not the Byzantine."""
        honest = [
            torch.tensor([0.0]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([1.5]),
            torch.tensor([2.0]),
            torch.tensor([2.5]),
        ]
        byzantine = torch.tensor([100.0])
        grads = honest + [byzantine]
        result = Bulyan.aggregate(grads, n=7, f=1, m=4)
        self.assertEqual(result.shape, (1,))
        self.assertLess(result.item(), 10.0)

    def test_check_rejects_m_below_one(self) -> None:
        """Check raises ValueError when m < 1."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate([torch.tensor([1.0])], n=7, f=1, m=0)

    def test_check_rejects_invalid_m(self) -> None:
        """Check raises ValueError when m is out of bounds."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=7, f=1, m=10)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=7, f=1)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when gradients.shape[0] != n."""
        agg = Bulyan(n=7, f=1)
        with self.assertRaises(ValueError):
            agg.aggregate(torch.tensor([[1.0], [2.0], [3.0]]))


if __name__ == "__main__":
    unittest.main()
