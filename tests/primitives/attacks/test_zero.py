"""Tests for zero attacks."""

import unittest

import torch

from krum.primitives.attacks import ZeroAttack


class ZeroAttackTest(unittest.TestCase):
    """Test ZeroAttack."""

    def test_generates_zero_gradients(self) -> None:
        """Attack returns one zero gradient per Byzantine worker."""
        honest_gradients = torch.ones((3, 5), dtype=torch.float64)

        byzantine_gradients = ZeroAttack()(honest_gradients, num_byzantine=2)

        self.assertEqual(byzantine_gradients.shape, (2, 5))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)
        self.assertTrue(torch.equal(byzantine_gradients, torch.zeros_like(byzantine_gradients)))


if __name__ == "__main__":
    unittest.main()
