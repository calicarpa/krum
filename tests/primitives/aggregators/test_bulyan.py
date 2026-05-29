"""Tests for the Bulyan aggregator."""

import unittest

import torch

from krum.primitives.aggregators import Bulyan


class BulyanTest(unittest.TestCase):
    """Test Bulyan aggregator."""

    def test_aggregate_returns_gradient(self) -> None:
        """Bulyan returns a gradient of the expected shape."""
        agg = Bulyan(n=7, f=1, m=4)
        grads = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ])
        result = agg.aggregate(grads)
        self.assertEqual(result.shape, (2,))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = Bulyan(n=7, f=1, m=4)
        grads = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
            dtype=torch.float64,
        )
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_default_m(self) -> None:
        """M defaults to n - f - 2 when not provided."""
        agg = Bulyan(n=7, f=1)
        self.assertEqual(agg.m, 4)

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
            Bulyan(n=5, f=1)

    def test_check_rejects_invalid_m(self) -> None:
        """Check raises ValueError when m is out of bounds."""
        with self.assertRaises(ValueError):
            Bulyan(n=7, f=1, m=10)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            Bulyan(7, 1)


if __name__ == "__main__":
    unittest.main()
