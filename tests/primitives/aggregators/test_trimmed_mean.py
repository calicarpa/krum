"""Tests for the Trimmed Mean aggregator."""

import unittest

import torch

from krum.primitives.aggregators.trimmed_mean import TrimmedMean


class TrimmedMeanTest(unittest.TestCase):
    """Test Trimmed Mean aggregator."""

    def test_aggregate_computes_coordinate_wise_trimmed_mean(self) -> None:
        """Aggregate returns the coordinate-wise trimmed mean."""
        grads = [
            torch.tensor([1.0, 10.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([3.0, 5.0]),
            torch.tensor([4.0, 7.0]),
            torch.tensor([100.0, 1.0]),
        ]
        result = TrimmedMean.aggregate(grads, f=1)
        expected = torch.tensor([3.0, 5.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_aggregate_odd_number_of_gradients(self) -> None:
        """Aggregate handles an odd number of gradients."""
        grads = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor([4.0]),
            torch.tensor([5.0]),
            torch.tensor([6.0]),
            torch.tensor([7.0]),
        ]
        result = TrimmedMean.aggregate(grads, f=2)
        self.assertAlmostEqual(result.item(), 4.0)

    def test_aggregate_all_f_trimmed(self) -> None:
        """Aggregate trims f outliers from both ends."""
        grads = [
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor([100.0]),
        ]
        result = TrimmedMean.aggregate(grads, f=2)
        self.assertAlmostEqual(result.item(), 2.0)

    def test_aggregate_preserves_dtype(self) -> None:
        """Aggregate preserves the input dtype."""
        grads = [
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            torch.tensor([5.0, 6.0], dtype=torch.float64),
            torch.tensor([7.0, 8.0], dtype=torch.float64),
            torch.tensor([9.0, 10.0], dtype=torch.float64),
        ]
        result = TrimmedMean.aggregate(grads, f=1)
        self.assertEqual(result.dtype, torch.float64)

    def test_check_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        with self.assertRaises(ValueError):
            TrimmedMean.aggregate([torch.tensor([1.0])], f=-1)

    def test_check_rejects_insufficient_gradients(self) -> None:
        """Check raises ValueError when len(gradients) <= 2*f."""
        with self.assertRaises(ValueError):
            TrimmedMean.aggregate(
                [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0])], f=2
            )

    def test_call_class_delegates_to_aggregate_with_kwargs(self) -> None:
        """Calling the class directly delegates to aggregate, including keyword args."""
        grads = [
            torch.tensor([1.0, 10.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([3.0, 5.0]),
            torch.tensor([4.0, 7.0]),
            torch.tensor([100.0, 1.0]),
        ]
        result = TrimmedMean(grads, f=1)
        expected = torch.tensor([3.0, 5.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_call_instance_delegates_to_aggregate_with_kwargs(self) -> None:
        """__call__ on an instance delegates to aggregate, including keyword args."""
        grads = [
            torch.tensor([1.0, 10.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([3.0, 5.0]),
            torch.tensor([4.0, 7.0]),
            torch.tensor([100.0, 1.0]),
        ]
        instance = TrimmedMean()
        result = instance(grads, f=1)
        expected = torch.tensor([3.0, 5.0])
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
