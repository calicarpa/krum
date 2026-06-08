"""Tests for the Gaussian attack."""

import unittest

import torch

from krum.primitives.attacks.gaussian import GaussianAttack


class GaussianAttackTest(unittest.TestCase):
    """Test Gaussian attack."""

    def test_generate_returns_correct_shape(self) -> None:
        """Generate returns a tensor of the expected shape."""
        attack = GaussianAttack(std=200.0)
        honest = torch.randn(5, 10)
        byz = attack.generate(honest, 3)
        self.assertEqual(byz.shape, (3, 10))

    def test_generate_zero_byzantine(self) -> None:
        """Generate with zero Byzantine workers returns an empty tensor."""
        attack = GaussianAttack(std=200.0)
        honest = torch.randn(5, 10)
        byz = attack.generate(honest, 0)
        self.assertEqual(byz.shape, (0, 10))

    def test_generate_preserves_dtype(self) -> None:
        """Generate preserves the input dtype."""
        attack = GaussianAttack(std=200.0)
        honest = torch.randn(5, 10, dtype=torch.float64)
        byz = attack.generate(honest, 3)
        self.assertEqual(byz.dtype, torch.float64)

    def test_generate_preserves_device(self) -> None:
        """Generate preserves the input device."""
        attack = GaussianAttack(std=200.0)
        honest = torch.randn(5, 10)
        byz = attack.generate(honest, 3)
        self.assertEqual(byz.device, honest.device)

    def test_generate_has_zero_mean_approximately(self) -> None:
        """Generated gradients have approximately zero mean when mu=0."""
        attack = GaussianAttack(mu=0.0, std=200.0)
        honest = torch.randn(10, 100)
        byz = attack.generate(honest, 1000)
        mean = byz.mean()
        self.assertLess(abs(mean.item()), 20.0)  # within ~3 sigma of sample mean

    def test_generate_respects_mu(self) -> None:
        """Generated gradients respect the configured mu."""
        attack = GaussianAttack(mu=42.0, std=0.0)
        honest = torch.randn(5, 10)
        byz = attack.generate(honest, 3)
        self.assertTrue(torch.allclose(byz, torch.full((3, 10), 42.0)))

    def test_default_mu_is_zero(self) -> None:
        """Mu defaults to 0."""
        attack = GaussianAttack(std=200.0)
        self.assertEqual(attack.mu, 0.0)

    def test_parameters_are_keyword_only(self) -> None:
        """Parameters must be passed as keywords."""
        with self.assertRaises(TypeError):
            GaussianAttack(200.0)  # ty:ignore[too-many-positional-arguments]

    def test_rejects_negative_std(self) -> None:
        """Check raises ValueError when std < 0."""
        with self.assertRaises(ValueError):
            GaussianAttack(std=-1.0)


if __name__ == "__main__":
    unittest.main()
