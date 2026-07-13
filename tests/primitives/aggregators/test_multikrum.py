"""Tests for the MultiKrum aggregator."""

import unittest

import torch

from krum.primitives.aggregators.multikrum import MultiKrum


class MultiKrumTest(unittest.TestCase):
    """Test MultiKrum aggregator."""

    def test_aggregate_averages_top_m(self) -> None:
        """MultiKrum averages the m gradients with smallest Krum scores."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [100.0, 100.0]])
        result = MultiKrum.aggregate(grads, n=7, f=1, m=2)
        self.assertEqual(result.shape, (2,))
        expected = torch.tensor([2.5, 0.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_m_equals_one_is_krum(self) -> None:
        """MultiKrum with m=1 is equivalent to Krum."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]])
        result = MultiKrum.aggregate(grads, n=5, f=1, m=1)
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 0.0])))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [100.0, 100.0]],
            dtype=torch.float64,
        )
        result = MultiKrum.aggregate(grads, n=7, f=1, m=2)
        self.assertEqual(result.dtype, torch.float64)

    def test_aggregate_at_max_m(self) -> None:
        """MultiKrum supports m at the upper bound n - 2f - 3."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [100.0, 100.0]])
        result = MultiKrum.aggregate(grads, n=7, f=1, m=2)
        expected = torch.tensor([2.5, 0.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_high_dimensional(self) -> None:
        """MultiKrum works on high-dimensional gradients."""
        grads = torch.randn(9, 512)
        grads[0] += 50.0
        result = MultiKrum.aggregate(grads, n=9, f=2, m=2)
        self.assertEqual(result.shape, (512,))

    def test_aggregate_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [100.0, 100.0]])
        out = torch.empty(2, dtype=torch.float32)
        result = MultiKrum.aggregate(grads, out, n=7, f=1, m=2)
        self.assertIs(result, out)
        self.assertTrue(torch.allclose(result, torch.tensor([2.5, 0.0])))

    def test_aggregate_accepts_sequence_of_per_worker_vectors(self) -> None:
        """A sequence of 1-D vectors gives the same result as the stacked tensor."""
        as_tensor = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [100.0, 100.0],
        ])
        as_sequence = [as_tensor[i] for i in range(as_tensor.shape[0])]
        self.assertTrue(
            torch.allclose(
                MultiKrum.aggregate(as_sequence, n=7, f=1, m=2), MultiKrum.aggregate(as_tensor, n=7, f=1, m=2)
            )
        )

    def test_score_without_mask_matches_internal_helper(self) -> None:
        """MultiKrum.score with no mask returns the Krum scores for all workers."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]])
        scores = MultiKrum.score(grads, n=5, f=1)
        self.assertEqual(scores.shape, (5,))
        # The outlier (worker 4) should have the highest score
        self.assertEqual(int(scores.argmax().item()), 4)
        # Edge workers have higher scores than central ones
        self.assertLess(scores[1].item(), scores[0].item())
        self.assertLess(scores[2].item(), scores[0].item())
        self.assertLess(scores[1].item(), scores[3].item())
        self.assertLess(scores[2].item(), scores[3].item())

    def test_score_with_mask_excludes_workers(self) -> None:
        """MultiKrum.score with valid_mask treats masked workers as infinitely far."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]])
        mask = torch.tensor([True, True, True, True, False])
        scores = MultiKrum.score(grads, n=5, f=1, valid_mask=mask)
        # Worker 4 (masked) should have an infinite score
        self.assertEqual(float(scores[4]), float("inf"))
        # The remaining workers should still have finite scores
        self.assertTrue(torch.isfinite(scores[:4]).all())

    def test_score_with_mask_all_false_returns_inf(self) -> None:
        """MultiKrum.score with all workers masked returns all-infinite scores."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]])
        mask = torch.zeros(5, dtype=torch.bool)
        scores = MultiKrum.score(grads, n=5, f=1, valid_mask=mask)
        self.assertTrue(torch.isinf(scores).all())

    def test_check_rejects_non_int_n(self) -> None:
        """N must be an int."""
        with self.assertRaises(TypeError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n="5", f=1, m=1)  # ty: ignore[invalid-argument-type]

    def test_check_rejects_non_int_f(self) -> None:
        """F must be an int."""
        with self.assertRaises(TypeError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=5, f="1", m=1)  # ty: ignore[invalid-argument-type]

    def test_check_rejects_non_int_m(self) -> None:
        """M must be an int."""
        with self.assertRaises(TypeError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=5, f=1, m="1")  # ty: ignore[invalid-argument-type]

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=0, f=0, m=1)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=5, f=-1, m=1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=5, f=10, m=1)

    def test_check_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 3."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=3, f=1, m=1)

    def test_check_rejects_invalid_m_too_small(self) -> None:
        """Check raises ValueError when m < 1."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=5, f=1, m=0)

    def test_check_rejects_invalid_m_too_large(self) -> None:
        """Check raises ValueError when m > n - f - 2."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0]]), n=5, f=1, m=3)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            MultiKrum.aggregate(torch.tensor([[1.0], [2.0], [3.0]]), n=5, f=1, m=2)


if __name__ == "__main__":
    unittest.main()
