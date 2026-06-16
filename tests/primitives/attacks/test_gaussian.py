"""Tests for the Gaussian attack."""

import unittest

import torch

from krum.primitives.attacks.gaussian import GaussianAttack


class GaussianAttackTest(unittest.TestCase):
    """Test Gaussian attack."""

    def test_generate_returns_correct_shape(self) -> None:
        """Generate returns a tensor of the expected shape."""
        honest = torch.randn(5, 10)
        byz = GaussianAttack.generate(honest, f=3)
        self.assertEqual(byz.shape, (3, 10))

    def test_generate_zero_byzantine(self) -> None:
        """Generate with zero Byzantine workers returns an empty tensor."""
        honest = torch.randn(5, 10)
        byz = GaussianAttack.generate(honest, f=0)
        self.assertEqual(byz.shape, (0, 10))

    def test_generate_preserves_dtype(self) -> None:
        """Generate preserves the input dtype."""
        honest = torch.randn(5, 10, dtype=torch.float64)
        byz = GaussianAttack.generate(honest, f=3)
        self.assertEqual(byz.dtype, torch.float64)

    def test_generate_preserves_device(self) -> None:
        """Generate preserves the input device."""
        honest = torch.randn(5, 10)
        byz = GaussianAttack.generate(honest, f=3)
        self.assertEqual(byz.device, honest.device)

    def test_generate_has_zero_mean_approximately(self) -> None:
        """Generated gradients have approximately zero mean when mu=0."""
        honest = torch.randn(10, 100)
        byz = GaussianAttack.generate(honest, f=1000, mu=0.0, std=200.0)
        mean = byz.mean()
        self.assertLess(abs(mean.item()), 20.0)

    def test_generate_respects_mu(self) -> None:
        """Generated gradients respect the configured mu."""
        honest = torch.randn(5, 10)
        byz = GaussianAttack.generate(honest, f=3, mu=42.0, std=0.0)
        self.assertTrue(torch.allclose(byz, torch.full((3, 10), 42.0)))

    def test_rejects_negative_std(self) -> None:
        """Check raises ValueError when std < 0."""
        honest = torch.randn(5, 10)
        with self.assertRaises(ValueError):
            GaussianAttack.generate(honest, f=3, std=-1.0)

    def test_rejects_negative_f(self) -> None:
        """Check raises ValueError when f < 0."""
        honest = torch.randn(5, 10)
        with self.assertRaises(ValueError):
            GaussianAttack.generate(honest, f=-1)

    def test_rejects_empty_honest_gradients(self) -> None:
        """Check raises ValueError when no honest gradients."""
        with self.assertRaises(ValueError):
            GaussianAttack.generate(torch.empty((0, 5)), f=3)

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_sequence = [
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([5.0, 7.0], dtype=torch.float64),
        ]
        as_tensor = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)

        from_sequence = GaussianAttack.generate(as_sequence, f=3, mu=0.0, std=1.0)
        from_tensor = GaussianAttack.generate(as_tensor, f=3, mu=0.0, std=1.0)

        self.assertEqual(from_sequence.shape, from_tensor.shape)
        self.assertEqual(from_sequence.dtype, from_tensor.dtype)

    def test_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        honest = torch.randn(5, 10, dtype=torch.float64)
        out = torch.empty((3, 10), dtype=torch.float64)

        result = GaussianAttack.generate(honest, out, f=3, mu=42.0, std=0.0)

        self.assertIs(result, out)
        self.assertTrue(torch.allclose(result, torch.full((3, 10), 42.0, dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
