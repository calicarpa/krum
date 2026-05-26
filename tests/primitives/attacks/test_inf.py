"""Tests for infinity attacks."""

import unittest

import torch

from krum.primitives.attacks import InfAttack


class InfAttackTest(unittest.TestCase):
    """Test InfAttack."""

    def test_generates_positive_infinite_gradients(self) -> None:
        """Attack returns one infinite gradient per Byzantine worker."""
        honest_gradients = torch.zeros((3, 5), dtype=torch.float64)

        byzantine_gradients = InfAttack()(honest_gradients, num_byzantine=2)

        self.assertEqual(byzantine_gradients.shape, (2, 5))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)
        self.assertTrue(torch.isposinf(byzantine_gradients).all())

    def test_generates_negative_infinite_gradients(self) -> None:
        """Attack can generate negative infinity."""
        honest_gradients = torch.zeros((3, 5))

        byzantine_gradients = InfAttack(sign="negative")(honest_gradients, num_byzantine=2)

        self.assertTrue(torch.isneginf(byzantine_gradients).all())

    def test_rejects_invalid_sign(self) -> None:
        """Attack sign must be positive or negative."""
        with self.assertRaises(ValueError):
            InfAttack(sign="zero")

    def test_sign_is_keyword_only(self) -> None:
        """Attack sign cannot be passed positionally."""
        with self.assertRaises(TypeError):
            InfAttack("negative")  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
