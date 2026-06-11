"""Tests for nearest-neighbor averaging."""

import unittest

import torch

from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage


class NearestNeighborAverageTest(unittest.TestCase):
    """Test nearest-neighbor averaging."""

    def test_aggregate_averages_closest_vectors_to_pivot(self) -> None:
        """Aggregate keeps the num_closest vectors nearest the pivot."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]])

        result = NearestNeighborAverage.aggregate(grads, num_closest=3, pivot=torch.tensor([0.0, 0.0]))

        expected = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_uses_call_specific_pivot(self) -> None:
        """The pivot is supplied for each aggregation call."""
        grads = torch.tensor([[100.0, 0.0], [11.0, 0.0], [10.0, 0.0], [9.0, 0.0]])

        result = NearestNeighborAverage.aggregate(grads, num_closest=3, pivot=torch.tensor([10.0, 0.0]))

        expected = torch.tensor([10.0, 0.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_pivot_does_not_need_to_be_a_candidate(self) -> None:
        """The pivot can be supplied independently of the candidate tensor."""
        grads = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]])

        result = NearestNeighborAverage.aggregate(grads, num_closest=2, pivot=torch.tensor([9.5, 0.0]))

        expected = torch.tensor([10.5, 0.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves input dtype."""
        grads = torch.tensor([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]], dtype=torch.float64)

        result = NearestNeighborAverage.aggregate(
            grads, num_closest=2, pivot=torch.tensor([0.0, 0.0], dtype=torch.float64)
        )

        self.assertEqual(result.dtype, torch.float64)

    def test_rejects_fewer_candidates_than_num_closest(self) -> None:
        """There must be at least num_closest candidates to average."""
        with self.assertRaises(ValueError):
            NearestNeighborAverage.aggregate(torch.empty((2, 4)), num_closest=3, pivot=torch.empty(4))

    def test_rejects_wrong_pivot_shape(self) -> None:
        """The pivot must have the same vector shape as a candidate."""
        with self.assertRaises(ValueError):
            NearestNeighborAverage.aggregate(torch.empty((3, 4)), num_closest=2, pivot=torch.empty(5))

    def test_rejects_non_positive_num_closest(self) -> None:
        """num_closest must be at least 1."""
        with self.assertRaises(ValueError):
            NearestNeighborAverage.aggregate(torch.empty((3, 4)), num_closest=0, pivot=torch.empty(4))

    def test_requires_pivot_keyword(self) -> None:
        """The pivot is required for each aggregation call."""
        with self.assertRaises(TypeError):
            NearestNeighborAverage.aggregate(torch.empty((3, 4)), num_closest=2)  # ty: ignore[missing-argument]

    def test_parameters_are_keyword_only(self) -> None:
        """num_closest and pivot must be passed as keywords, not positionally."""
        with self.assertRaises(TypeError):
            # gradients and out fill the only positional slots; num_closest cannot follow.
            NearestNeighborAverage.aggregate(torch.empty((3, 4)), torch.empty(4), 2)  # ty: ignore[too-many-positional-arguments, missing-argument]


if __name__ == "__main__":
    unittest.main()
