"""Tests for the AKSEL aggregator."""

import unittest

import torch

from krum.primitives.aggregators.aksel import Aksel


class AkselTest(unittest.TestCase):
    """Test AKSEL aggregator."""

    def test_aggregate_returns_mean_of_closest_to_median(self) -> None:
        """AKSEL selects the n-f gradients closest to the coordinate-wise median."""
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [100.0, 100.0]])
        result = Aksel.aggregate(grads, f=1)
        expected = torch.tensor([2.5, 2.5])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_identical_gradients(self) -> None:
        """AKSEL returns the common gradient when all are identical."""
        grads = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = Aksel.aggregate(grads, f=0)
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 2.0])))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [100.0, 100.0]], dtype=torch.float64)
        result = Aksel.aggregate(grads, f=1)
        self.assertEqual(result.dtype, torch.float64)

    def test_aggregate_f_zero_is_mean(self) -> None:
        """With f=0, AKSEL returns the ordinary coordinate-wise mean."""
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = Aksel.aggregate(grads, f=0)
        self.assertTrue(torch.equal(result, torch.tensor([3.0, 4.0])))

    def test_aggregate_high_dimensional(self) -> None:
        """AKSEL works on high-dimensional gradients."""
        grads = torch.randn(9, 256)
        grads[0] += 50.0
        result = Aksel.aggregate(grads, f=2)
        self.assertEqual(result.shape, (256,))

    def test_aggregate_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [100.0, 100.0]])
        out = torch.empty(2, dtype=torch.float32)
        result = Aksel.aggregate(grads, out, f=1)
        self.assertIs(result, out)
        self.assertTrue(torch.allclose(result, torch.tensor([2.5, 2.5])))

    def test_aggregate_accepts_sequence_of_per_worker_vectors(self) -> None:
        """A sequence of 1-D vectors gives the same result as the stacked tensor."""
        as_tensor = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [100.0, 100.0]])
        as_sequence = [as_tensor[i] for i in range(as_tensor.shape[0])]
        self.assertTrue(torch.equal(Aksel.aggregate(as_sequence, f=1), Aksel.aggregate(as_tensor, f=1)))

    def test_aggregate_ignores_byzantine_outliers(self) -> None:
        """AKSEL stays close to the honest cluster when outliers are present."""
        honest = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        byzantine = torch.tensor([[100.0, 100.0]])
        grads = torch.cat([honest, byzantine], dim=0)
        result = Aksel.aggregate(grads, f=1)
        self.assertLess(result[0].item(), 20.0)
        self.assertLess(result[1].item(), 20.0)

    def test_aggregate_matches_paper_for_tight_cluster(self) -> None:
        """On a tight honest cluster with outliers, AKSEL returns the cluster value."""
        honest = torch.full((9, 2), 1.0)
        byz = torch.tensor([[100.0, 100.0], [100.0, 100.0]])
        grads = torch.cat([honest, byz], dim=0)
        result = Aksel.aggregate(grads, f=2)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 1.0]), atol=1e-5))

    def test_aggregate_stays_in_honest_set(self) -> None:
        """The output is the mean of honest gradients, never a Byzantine value."""
        grads = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [100.0, 100.0],
            [100.0, 100.0],
        ])
        result = Aksel.aggregate(grads, f=2)
        self.assertTrue(torch.all(result < 10.0))

    def test_aggregate_minimal_valid_config(self) -> None:
        """AKSEL works with the smallest valid configuration n=2f+1."""
        grads = torch.tensor([[0.0], [0.1], [10.0]])
        result = Aksel.aggregate(grads, f=1)
        self.assertEqual(result.shape, (1,))

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            Aksel.aggregate(torch.tensor([[1.0]]), f=-1)

    def test_check_rejects_insufficient_gradients(self) -> None:
        """Check raises ValueError when len(gradients) <= 2*f."""
        with self.assertRaises(ValueError):
            Aksel.aggregate(torch.tensor([[1.0], [2.0], [3.0], [4.0]]), f=2)

    def test_aggregate_drops_extreme_outliers(self) -> None:
        """AKSEL removes extreme outliers and stays close to the honest mean."""
        grads = torch.tensor([
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [1000.0, 1000.0],
        ])
        result = Aksel.aggregate(grads, f=1)
        self.assertTrue(torch.all(result < 10.0))


if __name__ == "__main__":
    unittest.main()
