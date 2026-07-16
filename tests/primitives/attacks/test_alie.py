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

        # Algorithm 3: s = floor(n/2) + 1 - f, ratio = (n - s) / n where n = h + f
        h = 26
        n = h + f
        s = n // 2 + 1 - f
        ratio = (n - s) / n
        z_max = abs(
            torch.distributions.Normal(
                honest_gradients.new_tensor(0.0),
                honest_gradients.new_tensor(1.0),
            ).icdf(honest_gradients.new_tensor(ratio))
        )
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
            ALIEAttack.generate(honest_gradients, f=24, direction="zero")  # ty:ignore[invalid-argument-type]

    def test_factor_is_keyword_only(self) -> None:
        """Attack factor cannot be passed positionally."""
        honest_gradients = torch.zeros((26, 2), dtype=torch.float64)

        with self.assertRaises(TypeError):
            ALIEAttack.generate(honest_gradients, None, 1.0)  # ty:ignore[too-many-positional-arguments, missing-argument]

    def test_returns_empty_tensor_when_no_byzantine_gradients_are_requested(self) -> None:
        """Attack returns an empty tensor when no Byzantine gradients are requested."""
        honest_gradients = torch.zeros((3, 5), dtype=torch.float64)

        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=0)

        self.assertEqual(byzantine_gradients.shape, (0, 5))
        self.assertEqual(byzantine_gradients.dtype, honest_gradients.dtype)
        self.assertEqual(byzantine_gradients.device, honest_gradients.device)

    def test_rejects_worker_configuration_with_degenerate_target(self) -> None:
        """A worker configuration where ratio >= 1 raises ValueError."""
        # f=1, h=3 gives n=4, s=4//2+1-1=2, ratio=(4-2)/4=0.5 (valid)
        # f=1, h=1 gives n=2, s=2//2+1-1=1, ratio=(2-1)/2=0.5 (valid)
        # f=2, h=1 gives n=3, s=3//2+1-2=0, ratio=(3-0)/3=1.0 (invalid, ratio >= 1)
        honest_gradients = torch.zeros((1, 5))
        with self.assertRaises(ValueError):
            ALIEAttack.generate(honest_gradients, f=2)

    def test_rejects_worker_configuration_with_z_max_negative(self) -> None:
        """A worker configuration where z_max would be negative raises ValueError."""
        # f=3, h=2 gives n=5, s=5//2+1-3=0, ratio=(5-0)/5=1.0 (invalid)
        honest_gradients = torch.zeros((2, 5))
        with self.assertRaises(ValueError):
            ALIEAttack.generate(honest_gradients, f=3)

    def test_generates_positive_direction_with_paper_max_z(self) -> None:
        """The paper's max_z with POSITIVE direction puts the malicious gradient above the mean."""
        honest_gradients = torch.arange(52, dtype=torch.float64).reshape(26, 2)
        byzantine_gradients = ALIEAttack.generate(honest_gradients, f=24, direction=Direction.POSITIVE)
        h = 26
        n = h + 24
        s = n // 2 + 1 - 24
        ratio = (n - s) / n
        z_max = abs(
            torch.distributions.Normal(
                honest_gradients.new_tensor(0.0),
                honest_gradients.new_tensor(1.0),
            ).icdf(honest_gradients.new_tensor(ratio))
        )
        expected_gradient = honest_gradients.mean(dim=0) + z_max * honest_gradients.std(dim=0, correction=0)
        expected = expected_gradient.repeat(24, 1)
        self.assertTrue(torch.allclose(byzantine_gradients, expected))

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_tensor = torch.arange(52, dtype=torch.float64).reshape(26, 2)
        as_sequence = [as_tensor[i] for i in range(as_tensor.shape[0])]

        from_sequence = ALIEAttack.generate(as_sequence, f=24, z=1.0)
        from_tensor = ALIEAttack.generate(as_tensor, f=24, z=1.0)

        self.assertTrue(torch.equal(from_sequence, from_tensor))

    def test_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        honest_gradients = torch.arange(52, dtype=torch.float64).reshape(26, 2)
        out = torch.empty((24, 2), dtype=torch.float64)

        result = ALIEAttack.generate(honest_gradients, out, f=24, z=1.0)

        self.assertIs(result, out)
        self.assertTrue(torch.equal(result, ALIEAttack.generate(honest_gradients, f=24, z=1.0)))


if __name__ == "__main__":
    unittest.main()
