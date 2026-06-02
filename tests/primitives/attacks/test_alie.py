"""Tests for A Little Is Enough attacks."""

import unittest

import torch

from krum.primitives.attacks.alie import ALIEAttack, Direction


class ALIEAttackTest(unittest.TestCase):
    """Test ALIEAttack."""

    def test_generates_gradients_with_max_z(self) -> None:
        """Attack uses the maximal valid attack factor by default."""
        honest_gradients = torch.arange(52, dtype=torch.float64).reshape(26, 2)
        f = 24

        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=f)

        z_max = torch.distributions.Normal(
            honest_gradients.new_tensor(0.0),
            honest_gradients.new_tensor(1.0),
        ).icdf(honest_gradients.new_tensor(24 / 26))
        expected_gradient = honest_gradients.mean(dim=0) - z_max * honest_gradients.std(dim=0, correction=0)
        expected = expected_gradient.repeat(f, 1)

        self.assertEqual(byzantine_gradients.shape, (f, 2))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)
        self.assertTrue(torch.allclose(byzantine_gradients, expected))

    def test_generates_gradients_with_custom_z(self) -> None:
        """Attack can use an explicit valid attack factor."""
        honest_gradients = torch.arange(52, dtype=torch.float64).reshape(26, 2)

        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=24, z=1.0)

        expected_gradient = honest_gradients.mean(dim=0) - honest_gradients.std(dim=0, correction=0)
        expected = expected_gradient.repeat(24, 1)

        self.assertTrue(torch.allclose(byzantine_gradients, expected))

    def test_generates_positive_direction_gradients(self) -> None:
        """Attack can perturb in the positive direction."""
        honest_gradients = torch.arange(52, dtype=torch.float64).reshape(26, 2)

        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=24, z=1.0, direction=Direction.POSITIVE)

        expected_gradient = honest_gradients.mean(dim=0) + honest_gradients.std(dim=0, correction=0)
        expected = expected_gradient.repeat(24, 1)

        self.assertTrue(torch.allclose(byzantine_gradients, expected))

    def test_allows_attack_factor_above_z_max(self) -> None:
        """A numeric attack factor above z_max is used as-is."""
        honest_gradients = torch.arange(52, dtype=torch.float64).reshape(26, 2)

        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=24, z=2.0)

        expected_gradient = honest_gradients.mean(dim=0) - 2 * honest_gradients.std(dim=0, correction=0)
        expected = expected_gradient.repeat(24, 1)

        self.assertTrue(torch.allclose(byzantine_gradients, expected))

    def test_rejects_negative_z(self) -> None:
        """Attack factor must be non-negative."""
        honest_gradients = torch.zeros((26, 2), dtype=torch.float64)

        with self.assertRaises(ValueError):
            ALIEAttack.generate(honest_gradients, f=24, z=-1.0)

    def test_rejects_invalid_z(self) -> None:
        """Attack factor must be numeric or max."""
        honest_gradients = torch.zeros((26, 2), dtype=torch.float64)

        with self.assertRaises(TypeError):
            ALIEAttack.generate(honest_gradients, f=24, z="invalid")

    def test_rejects_invalid_direction(self) -> None:
        """Attack direction must be a Direction."""
        honest_gradients = torch.zeros((26, 2), dtype=torch.float64)

        with self.assertRaises(TypeError):
            ALIEAttack.generate(honest_gradients, f=24, direction="zero")  # type: ignore[arg-type]

    def test_factor_is_keyword_only(self) -> None:
        """Attack factor cannot be passed positionally."""
        honest_gradients = torch.zeros((26, 2), dtype=torch.float64)

        with self.assertRaises(TypeError):
            ALIEAttack.generate(honest_gradients, 1.0)  # type: ignore[misc]

    def test_returns_empty_tensor_when_no_byzantine_gradients_are_requested(self) -> None:
        """Attack returns an empty tensor when no Byzantine gradients are requested."""
        honest_gradients = torch.zeros((3, 5), dtype=torch.float64)

        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=0)

        self.assertEqual(byzantine_gradients.shape, (0, 5))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)

    def test_rejects_worker_configuration_without_non_negative_z_max(self) -> None:
        """Worker configuration must allow a non-negative maximal attack factor."""
        honest_gradients = torch.zeros((3, 5))

        with self.assertRaises(ValueError):
            ALIEAttack.generate(honest_gradients, f=1)

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_tensor = torch.arange(52, dtype=torch.float64).reshape(26, 2)
        as_sequence = [as_tensor[i] for i in range(as_tensor.shape[0])]

        from_sequence = ALIEAttack.generate(as_sequence, f=24, z=1.0)
        from_tensor = ALIEAttack.generate(as_tensor, f=24, z=1.0)

        self.assertTrue(torch.equal(from_sequence, from_tensor))


if __name__ == "__main__":
    unittest.main()
