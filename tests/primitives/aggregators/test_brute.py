"""Tests for the Brute aggregator."""

import unittest

import torch

from krum.primitives.aggregators.brute import Brute


class BruteTest(unittest.TestCase):
    """Test Brute aggregator."""

    def test_aggregate_selects_most_clumped_subset(self) -> None:
        """Brute selects the n-f subset with the smallest diameter and returns its mean."""
        grads = [
            torch.tensor([0.0]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([100.0]),
        ]
        result = Brute.aggregate(grads, n=4, f=1)
        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(result.item(), 0.5)

    def test_aggregate_multidimensional(self) -> None:
        """Brute works on multidimensional gradients."""
        grads = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([3.0, 0.0]),
            torch.tensor([100.0, 100.0]),
        ]
        result = Brute.aggregate(grads, n=5, f=1)
        self.assertEqual(result.shape, (2,))

    def test_aggregate_identical_gradients(self) -> None:
        """Brute returns the mean when all gradients are identical."""
        grads = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
        ]
        result = Brute.aggregate(grads, n=5, f=1)
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 2.0])))

    def test_aggregate_minimal_valid_config(self) -> None:
        """Brute works with the smallest valid configuration n=3, f=1."""
        grads = [
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.tensor([100.0]),
        ]
        result = Brute.aggregate(grads, n=3, f=1)
        self.assertEqual(result.shape, (1,))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = [
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64),
        ]
        result = Brute.aggregate(grads, n=4, f=1)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            Brute.aggregate([torch.tensor([1.0])], n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            Brute.aggregate([torch.tensor([1.0])], n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            Brute.aggregate([torch.tensor([1.0])], n=5, f=10)

    def test_check_rejects_zero_f(self) -> None:
        """Check raises ValueError when f = 0 (requires f >= 1)."""
        with self.assertRaises(ValueError):
            Brute.aggregate(
                [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
                n=3,
                f=0,
            )

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 1."""
        with self.assertRaises(ValueError):
            Brute.aggregate(
                [torch.tensor([1.0]), torch.tensor([2.0])],
                n=2,
                f=1,
            )

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            Brute.aggregate([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], n=5, f=1)


if __name__ == "__main__":
    unittest.main()
