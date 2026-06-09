"""Tests for the FullGradientNegation attack."""

import unittest

import torch

from krum.primitives.attacks.full_gradient_negation import FullGradientNegationAttack


class FullGradientNegationAttackTest(unittest.TestCase):
    """Test FullGradientNegation attack."""

    def test_generate_returns_correct_shape(self) -> None:
        """Generate returns a tensor of the expected shape."""
        honest = torch.randn(5, 10)
        full = torch.ones(10)
        byz = FullGradientNegationAttack.generate(honest, f=3, full_gradient=full)
        self.assertEqual(byz.shape, (3, 10))

    def test_generate_sends_negated_scaled_full_gradient(self) -> None:
        """Generated gradients equal -kappa * full_gradient."""
        honest = torch.randn(5, 3)
        full = torch.tensor([1.0, 2.0, 3.0])
        byz = FullGradientNegationAttack.generate(honest, f=2, full_gradient=full, kappa=100.0)
        expected = -100.0 * full
        self.assertTrue(torch.allclose(byz[0], expected))
        self.assertTrue(torch.allclose(byz[1], expected))

    def test_generate_zero_byzantine(self) -> None:
        """Generate with zero Byzantine workers returns an empty tensor."""
        honest = torch.randn(5, 10)
        full = torch.ones(10)
        byz = FullGradientNegationAttack.generate(honest, f=0, full_gradient=full)
        self.assertEqual(byz.shape, (0, 10))

    def test_generate_preserves_dtype(self) -> None:
        """Generate preserves the input dtype."""
        honest = torch.randn(5, 10, dtype=torch.float64)
        full = torch.ones(10, dtype=torch.float64)
        byz = FullGradientNegationAttack.generate(honest, f=3, full_gradient=full)
        self.assertEqual(byz.dtype, torch.float64)

    def test_rejects_negative_kappa(self) -> None:
        """Check raises ValueError when kappa < 0."""
        honest = torch.randn(5, 10)
        full = torch.ones(10)
        with self.assertRaises(ValueError):
            FullGradientNegationAttack.generate(honest, f=3, full_gradient=full, kappa=-1.0)

    def test_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        honest = torch.randn(5, 10)
        full = torch.ones(10)
        with self.assertRaises(ValueError):
            FullGradientNegationAttack.generate(honest, f=-1, full_gradient=full)

    def test_rejects_2d_full_gradient(self) -> None:
        """Check raises ValueError on 2D full gradient."""
        honest = torch.randn(5, 10)
        full = torch.ones(3, 10)
        with self.assertRaises(ValueError):
            FullGradientNegationAttack.generate(honest, f=3, full_gradient=full)

    def test_rejects_empty_honest_gradients(self) -> None:
        """Check raises ValueError when no honest gradients."""
        full = torch.ones(10)
        with self.assertRaises(ValueError):
            FullGradientNegationAttack.generate(torch.empty((0, 5)), f=3, full_gradient=full)

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_sequence = [
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([5.0, 7.0], dtype=torch.float64),
        ]
        as_tensor = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)
        full = torch.ones(2, dtype=torch.float64)

        from_sequence = FullGradientNegationAttack.generate(as_sequence, f=3, full_gradient=full)
        from_tensor = FullGradientNegationAttack.generate(as_tensor, f=3, full_gradient=full)

        self.assertTrue(torch.equal(from_sequence, from_tensor))

    def test_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        honest = torch.randn(5, 10, dtype=torch.float64)
        full = torch.ones(10, dtype=torch.float64)
        out = torch.empty((3, 10), dtype=torch.float64)

        result = FullGradientNegationAttack.generate(honest, out, f=3, full_gradient=full)

        self.assertIs(result, out)
        self.assertTrue(torch.allclose(result, FullGradientNegationAttack.generate(honest, f=3, full_gradient=full)))


if __name__ == "__main__":
    unittest.main()
