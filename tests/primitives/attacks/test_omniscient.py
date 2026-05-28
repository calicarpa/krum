"""Tests for the Omniscient attack."""

import unittest

import torch

from krum.primitives.attacks import OmniscientAttack


class OmniscientAttackTest(unittest.TestCase):
    """Test Omniscient attack."""

    def test_generate_returns_correct_shape(self) -> None:
        """Generate returns a tensor of the expected shape."""
        attack = OmniscientAttack(kappa=100.0)
        honest = torch.randn(5, 10)
        attack.set_full_gradient(torch.ones(10))
        byz = attack.generate(honest, 3)
        self.assertEqual(byz.shape, (3, 10))

    def test_generate_sends_negated_scaled_full_gradient(self) -> None:
        """Generated gradients equal -kappa * full_gradient."""
        attack = OmniscientAttack(kappa=100.0)
        honest = torch.randn(5, 3)
        full = torch.tensor([1.0, 2.0, 3.0])
        attack.set_full_gradient(full)
        byz = attack.generate(honest, 2)
        expected = -100.0 * full
        self.assertTrue(torch.allclose(byz[0], expected))
        self.assertTrue(torch.allclose(byz[1], expected))

    def test_generate_zero_byzantine(self) -> None:
        """Generate with zero Byzantine workers returns an empty tensor."""
        attack = OmniscientAttack(kappa=100.0)
        honest = torch.randn(5, 10)
        attack.set_full_gradient(torch.ones(10))
        byz = attack.generate(honest, 0)
        self.assertEqual(byz.shape, (0, 10))

    def test_generate_preserves_dtype(self) -> None:
        """Generate preserves the input dtype."""
        attack = OmniscientAttack(kappa=100.0)
        honest = torch.randn(5, 10, dtype=torch.float64)
        attack.set_full_gradient(torch.ones(10, dtype=torch.float64))
        byz = attack.generate(honest, 3)
        self.assertEqual(byz.dtype, torch.float64)

    def test_generate_raises_without_full_gradient(self) -> None:
        """Generate raises RuntimeError if full gradient not set."""
        attack = OmniscientAttack(kappa=100.0)
        honest = torch.randn(5, 10)
        with self.assertRaises(RuntimeError):
            attack.generate(honest, 3)

    def test_set_full_gradient_rejects_2d(self) -> None:
        """Set_full_gradient raises ValueError on 2D tensor."""
        attack = OmniscientAttack(kappa=100.0)
        with self.assertRaises(ValueError):
            attack.set_full_gradient(torch.ones(3, 10))

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            OmniscientAttack(100.0)

    def test_rejects_negative_kappa(self) -> None:
        """Check raises ValueError when kappa < 0."""
        with self.assertRaises(ValueError):
            OmniscientAttack(kappa=-1.0)


if __name__ == "__main__":
    unittest.main()
