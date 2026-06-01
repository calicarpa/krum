"""Tests for the Average aggregator."""

import unittest

import torch

from krum.primitives.aggregators.average import Average


class AverageTest(unittest.TestCase):
    """Test Average aggregator."""

    def test_aggregate_computes_mean(self) -> None:
        """Aggregate returns the coordinate-wise mean."""
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = Average.aggregate(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_single_gradient(self) -> None:
        """Aggregate with a single gradient returns it unchanged."""
        grads = [torch.tensor([7.0, 8.0, 9.0])]
        result = Average.aggregate(grads)
        self.assertTrue(torch.equal(result, grads[0]))

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = [
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            torch.tensor([5.0, 6.0], dtype=torch.float64),
        ]
        result = Average.aggregate(grads)
        self.assertEqual(result.dtype, torch.float64)

    def test_call_class_delegates_to_aggregate(self) -> None:
        """Calling the class directly delegates to aggregate."""
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = Average(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))

    def test_call_instance_delegates_to_aggregate(self) -> None:
        """__call__ on an instance delegates to aggregate."""
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        instance = Average()
        result = instance(grads)
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
