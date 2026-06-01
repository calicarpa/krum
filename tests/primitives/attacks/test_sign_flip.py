"""Tests for sign-flip attacks."""

import unittest

import torch

from krum.primitives.attacks.sign_flip import SignFlipAttack


class SignFlipAttackTest(unittest.TestCase):
    """Test SignFlipAttack."""

    def test_generates_sign_flipped_honest_mean(self) -> None:
        """Attack returns one sign-flipped honest mean per Byzantine worker."""
        honest_gradients = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        byzantine_gradients = SignFlipAttack()(honest_gradients, num_byzantine=3)

        expected = torch.tensor([[-3.0, -5.0], [-3.0, -5.0], [-3.0, -5.0]], dtype=torch.float64)
        self.assertEqual(byzantine_gradients.shape, (3, 2))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)
        self.assertTrue(torch.equal(byzantine_gradients, expected))

    def test_generates_scaled_sign_flipped_honest_mean(self) -> None:
        """Attack can scale the sign-flipped honest mean."""
        honest_gradients = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        byzantine_gradients = SignFlipAttack(scale=2.0)(honest_gradients, num_byzantine=2)

        expected = torch.tensor([[-6.0, -10.0], [-6.0, -10.0]], dtype=torch.float64)
        self.assertTrue(torch.equal(byzantine_gradients, expected))

    def test_rejects_negative_scale(self) -> None:
        """Attack scale must be non-negative."""
        with self.assertRaises(ValueError):
            SignFlipAttack(scale=-1.0)

    def test_scale_is_keyword_only(self) -> None:
        """Attack scale cannot be passed positionally."""
        with self.assertRaises(TypeError):
            SignFlipAttack(2.0)  # type: ignore[misc]

    def test_rejects_empty_honest_gradients(self) -> None:
        """Attack needs at least one honest gradient to compute the honest mean."""
        with self.assertRaises(ValueError):
            SignFlipAttack()(torch.empty((0, 5)), num_byzantine=2)


if __name__ == "__main__":
    unittest.main()
