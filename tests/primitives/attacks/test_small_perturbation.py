"""Tests for the SmallPerturbation attack."""

import math
import unittest

import torch

from krum.primitives.aggregators.brute import Brute
from krum.primitives.attacks.small_perturbation import SmallPerturbationAttack


class SmallPerturbationAttackTest(unittest.TestCase):
    """Test SmallPerturbation attack."""

    def test_generate_returns_correct_shape(self) -> None:
        """Generate returns a tensor of the expected shape."""
        honest = torch.randn(4, 10)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5)
        self.assertEqual(byz.shape, (1, 10))

    def test_generate_preserves_dtype(self) -> None:
        """Generate preserves the input dtype."""
        honest = torch.randn(4, 10, dtype=torch.float64)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5)
        self.assertEqual(byz.dtype, torch.float64)

    def test_generate_preserves_device(self) -> None:
        """Generate preserves the input device."""
        honest = torch.randn(4, 10)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5)
        self.assertEqual(byz.device, honest.device)

    def test_generate_all_byzantine_same(self) -> None:
        """All Byzantine gradients are identical."""
        honest = torch.randn(4, 10)
        byz = SmallPerturbationAttack.generate(honest, f=2, aggregator=Brute, n=6)
        self.assertTrue(torch.equal(byz[0], byz[1]))

    def test_rejects_invalid_n(self) -> None:
        """Check raises ValueError when n < 1."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=0)

    def test_rejects_invalid_f(self) -> None:
        """Check raises ValueError when f < 1."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=0, aggregator=Brute, n=5)

    def test_rejects_f_greater_than_n(self) -> None:
        """Check raises ValueError when f > n."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=10, aggregator=Brute, n=5)

    def test_rejects_insufficient_workers(self) -> None:
        """Check raises ValueError when n < 2f + 1."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=2, aggregator=Brute, n=4)

    def test_rejects_invalid_norm(self) -> None:
        """Check raises ValueError when p is not 2 or inf."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, p=3)

    def test_rejects_invalid_coordinate_string(self) -> None:
        """Check raises ValueError when coordinate is invalid string."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, coordinate="invalid")

    def test_rejects_coordinate_all_with_finite_norm(self) -> None:
        """Check raises ValueError when coordinate='all' with p != inf."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, p=2, coordinate="all")

    def test_rejects_wrong_honest_count(self) -> None:
        """Check raises ValueError when honest count doesn't match n-f."""
        honest = torch.randn(3, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5)

    def test_rejects_empty_honest_gradients(self) -> None:
        """Check raises ValueError when no honest gradients."""
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(torch.empty((0, 5)), f=1, aggregator=Brute, n=5)

    def test_accepts_sequence_of_per_worker_vectors(self) -> None:
        """Honest gradients may be a sequence of 1-D vectors, not just a 2-D tensor."""
        as_sequence = [torch.randn(10) for _ in range(4)]
        as_tensor = torch.stack(as_sequence)

        from_sequence = SmallPerturbationAttack.generate(as_sequence, f=1, aggregator=Brute, n=5)
        from_tensor = SmallPerturbationAttack.generate(as_tensor, f=1, aggregator=Brute, n=5)

        self.assertTrue(torch.equal(from_sequence, from_tensor))

    def test_writes_into_out_buffer_and_returns_it(self) -> None:
        """A provided out buffer receives the result and is returned."""
        honest = torch.randn(4, 10, dtype=torch.float64)
        out = torch.empty((1, 10), dtype=torch.float64)

        result = SmallPerturbationAttack.generate(honest, out, f=1, aggregator=Brute, n=5)

        self.assertIs(result, out)

    def test_infinite_norm_uses_all_ones_direction(self) -> None:
        """With p=inf, the attack uses all-ones direction."""
        honest = torch.randn(4, 10)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, p=math.inf, coordinate="all")
        self.assertEqual(byz.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
