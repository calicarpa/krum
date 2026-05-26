"""Tests for the MultiKrum aggregator."""

import unittest

import torch

from krum.primitives.aggregators import MultiKrum


class MultiKrumTest(unittest.TestCase):
    """Test MultiKrum aggregator."""

    def test_aggregate_averages_top_m(self) -> None:
        """MultiKrum averages the m gradients with smallest Krum scores."""
        agg = MultiKrum(n=5, f=1, m=2)
        grads = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [100.0, 100.0],
        ])
        result = agg.aggregate(grads)
        self.assertEqual(result.shape, (2,))
        expected = torch.tensor([1.5, 0.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_m_equals_one_is_krum(self) -> None:
        """MultiKrum with m=1 is equivalent to Krum."""
        agg = MultiKrum(n=5, f=1, m=1)
        grads = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [100.0, 100.0],
        ])
        result = agg.aggregate(grads)
        self.assertTrue(torch.equal(result, grads[1]))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        agg = MultiKrum(n=5, f=1, m=2)
        grads = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]],
            dtype=torch.float64,
        )
        result = agg.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_influence_ratio_partial(self) -> None:
        """Influence_ratio returns the fraction of selected Byzantine gradients."""
        agg = MultiKrum(n=5, f=1, m=2)
        honest = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [200.0, 200.0]])
        byzantine = torch.tensor([[5.0, 0.0]])
        result = agg.influence_ratio(honest, byzantine)
        self.assertEqual(result, 0.5)

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

    def test_upper_bound_returns_finite_value(self) -> None:
        """upper_bound returns a finite positive value."""
        agg = MultiKrum(n=5, f=1, m=2)
        bound = agg.upper_bound()
        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 1.0)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            MultiKrum(5, 1, 2)


if __name__ == "__main__":
    unittest.main()
