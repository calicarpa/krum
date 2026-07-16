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

    def test_aggregate_with_byzantine_in_minority(self) -> None:
        """Bulyan ignores isolated outliers and stays close to the honest cluster."""
        honest = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        byzantine = torch.tensor([[100.0, 100.0]])
        grads = torch.cat([honest, byzantine], dim=0)
        result = Bulyan.aggregate(grads, n=7, f=1, m=4)
        self.assertEqual(result.shape, (2,))
        self.assertLess(result[0].item(), 20.0)
        self.assertLess(result[1].item(), 20.0)

    def test_aggregate_returns_shape_with_minimal_valid_config(self) -> None:
        """Bulyan works with the smallest valid configuration n=4f+3."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        result = Bulyan.aggregate(grads, n=7, f=1, m=1)
        self.assertEqual(result.shape, (2,))

    def test_aggregate_high_dimensional(self) -> None:
        """Bulyan works on high-dimensional gradients."""
        grads = torch.randn(11, 256)
        grads[0] += 50.0
        grads[1] += 50.0
        result = Bulyan.aggregate(grads, n=11, f=2, m=4)
        self.assertEqual(result.shape, (256,))

    def test_aggregate_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        out = torch.empty(2, dtype=torch.float32)
        result = Bulyan.aggregate(grads, out, n=7, f=1, m=4)
        self.assertIs(result, out)

    def test_aggregate_accepts_sequence_of_per_worker_vectors(self) -> None:
        """A sequence of 1-D vectors gives the same result as the stacked tensor."""
        as_tensor = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        as_sequence = [as_tensor[i] for i in range(as_tensor.shape[0])]
        self.assertTrue(
            torch.equal(Bulyan.aggregate(as_sequence, n=7, f=1, m=4), Bulyan.aggregate(as_tensor, n=7, f=1, m=4))
        )

    def test_aggregate_matches_paper_for_tight_cluster(self) -> None:
        """On a tight honest cluster with two outliers, Bulyan returns the cluster value."""
        honest = torch.full((9, 2), 1.0)
        byz = torch.tensor([[100.0, 100.0], [100.0, 100.0]])
        grads = torch.cat([honest, byz], dim=0)
        result = Bulyan.aggregate(grads, n=11, f=2, m=7)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 1.0]), atol=1e-5))

    def test_aggregate_stays_in_honest_set(self) -> None:
        """The output stays in the honest range, never a Byzantine value."""
        grads = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [7.0, 0.0],
            [8.0, 0.0],
            [100.0, 100.0],
            [100.0, 100.0],
        ])
        result = Bulyan.aggregate(grads, n=11, f=2, m=7)
        self.assertTrue(torch.all(result < 10.0))

    def test_aggregate_with_default_m_uses_n_minus_f(self) -> None:
        """When m is omitted, the default n-f is used (m=9 for n=11, f=2)."""
        grads = torch.randn(11, 4)
        with_explicit = Bulyan.aggregate(grads, n=11, f=2, m=9)
        with_default = Bulyan.aggregate(grads, n=11, f=2)
        self.assertTrue(torch.equal(with_explicit, with_default))

    def test_aggregate_recovers_honest_mean_when_no_outliers(self) -> None:
        """With all honest gradients, Bulyan's output stays in the central region of the cluster."""
        grads = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [4.0, 7.0], [5.0, 8.0], [6.0, 9.0], [7.0, 10.0]])
        result = Bulyan.aggregate(grads, n=7, f=1, m=4)
        self.assertTrue(2.0 < result[0].item() < 6.0)
        self.assertTrue(5.0 < result[1].item() < 9.0)

    def test_check_rejects_non_int_n(self) -> None:
        """N must be an int."""
        with self.assertRaises(TypeError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n="7", f=1, m=4)  # ty: ignore[invalid-argument-type]

    def test_check_rejects_non_int_f(self) -> None:
        """F must be an int."""
        with self.assertRaises(TypeError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=7, f="1", m=4)  # ty: ignore[invalid-argument-type]

    def test_check_rejects_non_int_m(self) -> None:
        """M must be an int."""
        with self.assertRaises(TypeError):
            Bulyan.aggregate(torch.tensor([[1.0]]), n=7, f=1, m="4")  # ty: ignore[invalid-argument-type]

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


if __name__ == "__main__":
    unittest.main()
