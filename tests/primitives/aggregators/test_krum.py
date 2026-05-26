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

    def test_influence_ratio_no_byzantine_influence(self) -> None:
        """influence_ratio is zero when Byzantine grads are far from the cluster."""
        agg = Krum(n=5, f=1)
        honest = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        byzantine = torch.tensor([[100.0, 100.0]])
        result = agg.influence_ratio(honest, byzantine)
        self.assertEqual(result, 0.0)

    def test_influence_ratio_outlier_not_selected(self) -> None:
        """Influence_ratio is zero when the Byzantine gradient is a far outlier."""
        agg = Krum(n=5, f=1)
        honest = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        byzantine = torch.tensor([[100.0, 100.0]])
        result = agg.influence_ratio(honest, byzantine)
        self.assertEqual(result, 0.0)

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 3."""
        with self.assertRaises(ValueError):
            Krum(n=3, f=1)

    def test_upper_bound_returns_finite_value(self) -> None:
        """upper_bound returns a finite positive value."""
        agg = Krum(n=5, f=1)
        bound = agg.upper_bound()
        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 1.0)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            Krum(5, 1)


if __name__ == "__main__":
    unittest.main()
