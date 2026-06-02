"""Tests for sign-flip attacks."""

import unittest

import torch

from krum.primitives.attacks.sign_flip import SignFlipAttack


class SignFlipAttackTest(unittest.TestCase):
    """Test SignFlipAttack."""

    def test_generates_sign_flipped_honest_mean(self) -> None:
        """Attack returns one sign-flipped honest mean per Byzantine worker."""
        honest_gradients = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        byzantine_gradients = SignFlipAttack.generate(honest_gradients, f=3)

        expected = torch.tensor([[-3.0, -5.0], [-3.0, -5.0], [-3.0, -5.0]], dtype=torch.float64)
        self.assertEqual(byzantine_gradients.shape, (3, 2))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)
        self.assertTrue(torch.equal(byzantine_gradients, expected))

    def test_generates_scaled_sign_flipped_honest_mean(self) -> None:
        """Attack can scale the sign-flipped honest mean."""
        honest_gradients = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        byzantine_gradients = SignFlipAttack.generate(honest_gradients, f=2, scale=2.0)

        expected = torch.tensor([[-6.0, -10.0], [-6.0, -10.0]], dtype=torch.float64)
        self.assertTrue(torch.equal(byzantine_gradients, expected))

    def test_rejects_negative_scale(self) -> None:
        """Attack scale must be non-negative."""
        honest_gradients = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        with self.assertRaises(ValueError):
            SignFlipAttack.generate(honest_gradients, f=2, scale=-1.0)

    def test_scale_is_keyword_only(self) -> None:
        """Attack scale cannot be passed positionally."""
        honest_gradients = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        with self.assertRaises(TypeError):
            SignFlipAttack.generate(honest_gradients, 2.0)  # type: ignore[misc]

    def test_rejects_empty_honest_gradients(self) -> None:
        """Attack needs at least one honest gradient to compute the honest mean."""
        with self.assertRaises(ValueError):
            SignFlipAttack.generate(torch.empty((0, 5)), f=2)

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_sequence = [
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([5.0, 7.0], dtype=torch.float64),
        ]
        as_tensor = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        from_sequence = SignFlipAttack.generate(as_sequence, f=3)
        from_tensor = SignFlipAttack.generate(as_tensor, f=3)

        self.assertTrue(torch.equal(from_sequence, from_tensor))


if __name__ == "__main__":
    unittest.main()
