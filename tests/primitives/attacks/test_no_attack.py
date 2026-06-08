"""Tests for the NoAttack."""

import unittest

import torch

from krum.primitives.attacks.no_attack import NoAttack


class NoAttackTest(unittest.TestCase):
    """Test NoAttack."""

    def test_generate_returns_empty_tensor(self) -> None:
        """Generate returns an empty tensor of shape (0, d)."""
        honest = torch.randn(5, 10)
        byz = NoAttack.generate(honest, f=3)
        self.assertEqual(byz.shape, (0, 10))

    def test_generate_preserves_dtype(self) -> None:
        """Generate preserves the input dtype."""
        honest = torch.randn(5, 10, dtype=torch.float64)
        byz = NoAttack.generate(honest, f=3)
        self.assertEqual(byz.dtype, torch.float64)

    def test_generate_preserves_device(self) -> None:
        """Generate preserves the input device."""
        honest = torch.randn(5, 10)
        byz = NoAttack.generate(honest, f=3)
        self.assertEqual(byz.device, honest.device)

    def test_generate_ignores_f(self) -> None:
        """Generate ignores f and always returns empty tensor."""
        honest = torch.randn(5, 10)
        byz = NoAttack.generate(honest, f=100)
        self.assertEqual(byz.shape, (0, 10))

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_sequence = [
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([5.0, 7.0], dtype=torch.float64),
        ]
        as_tensor = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        from_sequence = NoAttack.generate(as_sequence, f=3)
        from_tensor = NoAttack.generate(as_tensor, f=3)

        self.assertTrue(torch.equal(from_sequence, from_tensor))


if __name__ == "__main__":
    unittest.main()
