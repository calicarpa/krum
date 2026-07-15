"""Tests for the SmallPerturbation attack."""

import math
import unittest

import torch

from krum.primitives.aggregators import Aggregator
from krum.primitives.aggregators.brute import Brute
from krum.primitives.aggregators.multikrum import MultiKrum
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

    # ------------------------------------------------------------------
    # Regression tests for the boundary-search and selection-test bugs.
    # ------------------------------------------------------------------

    def test_default_attack_is_not_a_noop(self) -> None:
        """The default boundary search finds a non-zero gamma.

        Regression: previously the non-monotonic selection window sat
        above ``gamma_init`` so the search broke immediately and returned
        ``gamma = 0`` (the byzantine vector collapsed onto the honest
        mean — a no-op attack).
        """
        torch.manual_seed(0)
        honest = torch.randn(4, 10)
        honest_mean = honest.mean(dim=0)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5)
        self.assertGreater((byz[0] - honest_mean).abs().max().item(), 1.0)

    def test_byzantine_equals_honest_mean_plus_gamma_direction(self) -> None:
        """B(gamma) = honest_mean + gamma * E, perturbing only one coordinate."""
        torch.manual_seed(0)
        honest = torch.randn(4, 10)
        honest_mean = honest.mean(dim=0)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5)

        std = honest.std(dim=0, correction=0)
        chosen = int(torch.argmax(std).item())
        delta = byz[0] - honest_mean
        nonzero = delta.abs().nonzero().flatten().tolist()
        self.assertEqual(nonzero, [chosen])
        self.assertTrue(torch.allclose(delta[delta != 0], delta[delta != 0]))

    def test_explicit_gamma_bypasses_search(self) -> None:
        """Passing gamma= uses that exact value instead of searching."""
        honest = torch.randn(4, 10)
        honest_mean = honest.mean(dim=0)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, gamma=3.0)
        std = honest.std(dim=0, correction=0)
        chosen = int(torch.argmax(std).item())
        self.assertAlmostEqual(byz[0, chosen].item(), (honest_mean[chosen] + 3.0).item(), places=5)

    def test_integer_coordinate_is_used(self) -> None:
        """An explicit integer coordinate selects that coordinate only."""
        honest = torch.randn(4, 10)
        honest_mean = honest.mean(dim=0)
        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, gamma=3.0, coordinate=2)
        delta = byz[0] - honest_mean
        nonzero = delta.abs().nonzero().flatten().tolist()
        self.assertEqual(nonzero, [2])
        self.assertAlmostEqual(delta[2].item(), 3.0, places=5)

    def test_coordinate_largest_matches_default(self) -> None:
        """coordinate='largest' is an alias for the default max-variance choice."""
        honest = torch.randn(4, 10)
        default = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, gamma=2.0)
        largest = SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, gamma=2.0, coordinate="largest")
        self.assertTrue(torch.equal(default, largest))

    def test_infinite_norm_direction_is_all_ones_for_explicit_gamma(self) -> None:
        """p=inf perturbs every coordinate uniformly by gamma."""
        honest = torch.randn(4, 10)
        byz = SmallPerturbationAttack.generate(
            honest, f=1, aggregator=Brute, n=5, p=math.inf, coordinate="all", gamma=2.0
        )
        honest_mean = honest.mean(dim=0)
        delta = byz[0] - honest_mean
        self.assertTrue(torch.allclose(delta, torch.full((10,), 2.0, dtype=delta.dtype)))

    def test_rejects_coordinate_out_of_range(self) -> None:
        """An integer coordinate beyond d raises ValueError."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, coordinate=10)

    def test_rejects_non_positive_search_params(self) -> None:
        """gamma_max, gamma_init, and tol must be positive."""
        honest = torch.randn(4, 10)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, gamma_max=0.0)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, gamma_init=0.0)
        with self.assertRaises(ValueError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, tol=0.0)

    def test_rejects_non_int_coordinate_type(self) -> None:
        """A non int/str coordinate raises TypeError."""
        honest = torch.randn(4, 10)
        with self.assertRaises(TypeError):
            SmallPerturbationAttack.generate(honest, f=1, aggregator=Brute, n=5, coordinate=1.5)  # ty:ignore[invalid-argument-type]

    def test_is_selected_returns_python_bool(self) -> None:
        """_is_selected returns a real bool, not a 0-dim tensor."""
        honest = torch.randn(4, 10)
        honest_mean = honest.mean(dim=0)
        result = SmallPerturbationAttack._is_selected(honest, honest_mean, Brute, 5, 1, {})
        self.assertIsInstance(result, bool)

    def test_multikrum_target_finds_nontrivial_gamma(self) -> None:
        """The search also produces a non-trivial perturbation for MultiKrum."""
        torch.manual_seed(1)
        honest = torch.randn(6, 10)
        honest_mean = honest.mean(dim=0)
        byz = SmallPerturbationAttack.generate(honest, f=2, aggregator=MultiKrum, aggregator_kwargs={"m": 1}, n=8)
        self.assertGreater((byz[0] - honest_mean).abs().max().item(), 1.0)

    def test_search_returns_zero_when_never_selected(self) -> None:
        """An aggregator that ignores byzantine inputs yields gamma = 0 (no effect)."""
        honest = torch.randn(4, 10)

        class _IgnoreByz(Aggregator):
            @classmethod
            def aggregate(cls, gradients, /, out=None, *, n, f, **specialized):
                honest_only = gradients[: n - f]
                return honest_only.mean(dim=0)

        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=_IgnoreByz, n=5)
        self.assertTrue(torch.allclose(byz[0], honest.mean(dim=0)))

    def test_search_returns_gamma_max_when_never_rejected(self) -> None:
        """An aggregator that echoes byzantine inputs keeps B(gamma) selected forever."""
        honest = torch.randn(4, 10)

        class _EchoByz(Aggregator):
            @classmethod
            def aggregate(cls, gradients, /, out=None, *, n, f, **specialized):
                return gradients[n - f :].mean(dim=0)

        byz = SmallPerturbationAttack.generate(honest, f=1, aggregator=_EchoByz, n=5, gamma_max=1e3)
        delta = byz[0] - honest.mean(dim=0)
        # gamma clamped to gamma_max → perturbation magnitude equals gamma_max.
        self.assertAlmostEqual(delta.abs().max().item(), 1e3, places=0)


if __name__ == "__main__":
    unittest.main()
