"""Tests for the Median aggregator."""

import unittest

import torch

from krum.primitives.aggregators.median import Median


class MedianTest(unittest.TestCase):
    """Test Median aggregator."""

    def test_aggregate_computes_coordinate_wise_median(self) -> None:
        """Aggregate returns the coordinate-wise median."""
        grads = [torch.tensor([1.0, 9.0, 2.0]), torch.tensor([3.0, 1.0, 5.0]), torch.tensor([5.0, 4.0, 8.0])]
        result = Median.aggregate(grads)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_odd_number_of_gradients(self) -> None:
        """Aggregate handles an odd number of gradients."""
        grads = [
            torch.tensor([1.0]),
            torch.tensor([3.0]),
            torch.tensor([5.0]),
            torch.tensor([2.0]),
            torch.tensor([4.0]),
        ]
        result = Median.aggregate(grads)
        self.assertEqual(result.item(), 3.0)

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with one gradient returns it."""
        grads = [torch.tensor([7.0, 8.0])]
        result = Median.aggregate(grads)
        self.assertTrue(torch.equal(result, grads[0]))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = [
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            torch.tensor([5.0, 6.0], dtype=torch.float64),
        ]
        result = Median.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_call_class_delegates_to_aggregate(self) -> None:
        """Calling the class directly delegates to aggregate."""
        grads = [torch.tensor([1.0, 9.0, 2.0]), torch.tensor([3.0, 1.0, 5.0]), torch.tensor([5.0, 4.0, 8.0])]
        result = Median(grads)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_call_instance_delegates_to_aggregate(self) -> None:
        """__call__ on an instance delegates to aggregate."""
        grads = [torch.tensor([1.0, 9.0, 2.0]), torch.tensor([3.0, 1.0, 5.0]), torch.tensor([5.0, 4.0, 8.0])]
        instance = Median()
        result = instance(grads)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
