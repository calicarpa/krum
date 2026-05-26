"""Tests for NaN attacks."""

import unittest

import torch

from krum.primitives.attacks import NaNAttack


class NaNAttackTest(unittest.TestCase):
    """Test NaNAttack."""

    def test_generates_nan_gradients(self) -> None:
        """Attack returns one NaN gradient per Byzantine worker."""
        honest_gradients = torch.zeros((3, 5), dtype=torch.float64)

        byzantine_gradients = NaNAttack()(honest_gradients, num_byzantine=2)

        self.assertEqual(byzantine_gradients.shape, (2, 5))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)
        self.assertTrue(torch.isnan(byzantine_gradients).all())

    def test_rejects_non_2d_gradients(self) -> None:
        """Honest gradients must be a worker-by-parameter tensor."""
        with self.assertRaises(ValueError):
            NaNAttack()(torch.zeros(5), num_byzantine=2)

    def test_rejects_integer_gradients(self) -> None:
        """Honest gradients must be floating-point values."""
        with self.assertRaises(TypeError):
            NaNAttack()(torch.zeros((3, 5), dtype=torch.int64), num_byzantine=2)

    def test_rejects_negative_num_byzantine(self) -> None:
        """The number of Byzantine gradients cannot be negative."""
        with self.assertRaises(ValueError):
            NaNAttack()(torch.zeros((3, 5)), num_byzantine=-1)


if __name__ == "__main__":
    unittest.main()
