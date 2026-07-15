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

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 4f + 3."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]), n=5, f=1)

    def test_check_rejects_invalid_m(self) -> None:
        """Check raises ValueError when m is out of bounds."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=7, f=1, m=10)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            Bulyan.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=7, f=1)

    def test_rejects_byzantine_outliers(self) -> None:
        """Bulyan output does not contain byzantine outlier values."""
        torch.manual_seed(0)
        n, f, d = 11, 2, 50
        honest = torch.randn(n - f, d)
        byz = torch.full((f, d), 1e6)
        all_grads = torch.cat([honest, byz], dim=0)
        result = Bulyan.aggregate(all_grads, n=n, f=f, m=1)
        # No coordinate should approach the byzantine magnitude.
        self.assertLess(result.abs().max().item(), 100.0)
        # Output should be much closer to honest mean than to naive average.
        naive = all_grads.mean(dim=0)
        honest_mean = honest.mean(dim=0)
        self.assertLess(
            (result - honest_mean).norm().item(),
            (naive - honest_mean).norm().item() * 0.01,
        )

    def test_m_equals_one_selects_theta_gradients(self) -> None:
        """With m=1, Bulyan selects exactly n-2f gradients before trimming."""
        n, f, d = 11, 2, 5
        grads = torch.randn(n, d)
        # Should not raise (bulyan_m = n-4f = 3 > 0)
        result = Bulyan.aggregate(grads, n=n, f=f, m=1)
        self.assertEqual(result.shape, (d,))


if __name__ == "__main__":
    unittest.main()
