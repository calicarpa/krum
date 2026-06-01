"""Tests for the GeoMed aggregator."""

import unittest

import torch

from krum.primitives.aggregators.geomed import GeoMed


class GeoMedTest(unittest.TestCase):
    """Test GeoMed aggregator."""

    def test_aggregate_selects_geometric_median(self) -> None:
        """GeoMed selects the gradient minimizing the sum of distances to all others."""
        grads = [
            torch.tensor([0.0]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([100.0]),
        ]
        result = GeoMed.aggregate(grads, n=4, f=1)
        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(result.item(), 0.5)

    def test_aggregate_multidimensional(self) -> None:
        """GeoMed works on multidimensional gradients."""
        grads = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([10.0, 10.0]),
        ]
        result = GeoMed.aggregate(grads, n=4, f=1)
        self.assertEqual(result.shape, (2,))
        expected = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_identical_gradients(self) -> None:
        """GeoMed returns the first gradient when all are identical."""
        grads = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
        ]
        result = GeoMed.aggregate(grads, n=3, f=0)
        self.assertTrue(torch.equal(result, grads[0]))

    def test_aggregate_single_gradient(self) -> None:
        """GeoMed with a single gradient returns it unchanged."""
        grads = [torch.tensor([7.0, 8.0, 9.0])]
        result = GeoMed.aggregate(grads, n=1, f=0)
        self.assertTrue(torch.equal(result, grads[0]))

    def test_aggregate_ties_broken_by_smallest_index(self) -> None:
        """GeoMed breaks ties by returning the gradient with the smallest index."""
        grads = [
            torch.tensor([0.0]),
            torch.tensor([2.0]),
            torch.tensor([2.0]),
        ]
        result = GeoMed.aggregate(grads, n=3, f=0)
        self.assertTrue(torch.equal(result, grads[1]))

    def test_aggregate_f_is_ignored(self) -> None:
        """GeoMed accepts any valid f but does not use it."""
        grads = [
            torch.tensor([0.0]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([100.0]),
        ]
        result = GeoMed.aggregate(grads, n=4, f=1)
        self.assertAlmostEqual(result.item(), 0.5)

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = [
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64),
        ]
        result = GeoMed.aggregate(grads, n=4, f=1)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        with self.assertRaises(ValueError):
            GeoMed.aggregate([torch.tensor([1.0])], n=0, f=0)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            GeoMed.aggregate([torch.tensor([1.0])], n=5, f=-1)

    def test_check_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        with self.assertRaises(ValueError):
            GeoMed.aggregate([torch.tensor([1.0])], n=5, f=10)

    def test_check_rejects_wrong_number_of_gradients(self) -> None:
        """Check raises ValueError when len(gradients) != n."""
        with self.assertRaises(ValueError):
            GeoMed.aggregate([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], n=5, f=1)


if __name__ == "__main__":
    unittest.main()
