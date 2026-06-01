"""Tests for the MultiKrum aggregator."""

import unittest

import torch

from krum.primitives.aggregators.multikrum import MultiKrum


class MultiKrumTest(unittest.TestCase):
    """Test MultiKrum aggregator."""

    def test_aggregate_averages_top_m(self) -> None:
        """MultiKrum averages the m gradients with smallest Krum scores."""
        agg = MultiKrum(n=5, f=1, m=2)
        grads = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([3.0, 0.0]),
            torch.tensor([100.0, 100.0]),
        ]
        result = agg.aggregate(grads)
        self.assertEqual(result.shape, (2,))
        expected = torch.tensor([1.5, 0.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_m_equals_one_is_krum(self) -> None:
        """MultiKrum with m=1 is equivalent to Krum."""
        agg = MultiKrum(n=5, f=1, m=1)
        grads = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([3.0, 0.0]),
            torch.tensor([100.0, 100.0]),
        ]
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads[1]))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = MultiKrum(n=5, f=1, m=2)
        grads = [
            torch.tensor([0.0, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([2.0, 0.0], dtype=torch.float64),
            torch.tensor([3.0, 0.0], dtype=torch.float64),
            torch.tensor([100.0, 100.0], dtype=torch.float64),
        ]
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            MultiKrum(n=0, f=0, m=1)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            MultiKrum(n=5, f=-1, m=1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            MultiKrum(n=5, f=10, m=1)

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 3."""
        with self.assertRaises(ValueError):
            MultiKrum(n=3, f=1, m=1)

    def test_check_rejects_invalid_m_too_small(self) -> None:
        """Check raises ValueError when m < 1."""
        with self.assertRaises(ValueError):
            MultiKrum(n=5, f=1, m=0)

    def test_check_rejects_invalid_m_too_large(self) -> None:
        """Check raises ValueError when m > n - f - 2."""
        with self.assertRaises(ValueError):
            MultiKrum(n=5, f=1, m=3)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            MultiKrum(5, 1, 2)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        agg = MultiKrum(n=5, f=1, m=2)
        with self.assertRaises(ValueError):
            agg.aggregate([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])])


if __name__ == "__main__":
    unittest.main()
