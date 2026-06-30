"""Tests for CentralisedSimulation."""

import unittest
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from krum.primitives.aggregators.average import Average
from krum.simulations.centralised import CentralisedSimulation


class _DummyModel(nn.Module):
    """Simple linear classifier for testing."""

    def __init__(self, in_dim: int = 10, out_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the linear layer."""
        return self.linear(x)


def _dummy_dataset(n: int = 32, d: int = 10, classes: int = 2) -> TensorDataset:
    """Return a synthetic classification dataset with ``n`` samples."""
    x = torch.randn(n, d)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(x, y)


class CentralisedSimulationConstructionTest(unittest.TestCase):
    """Test construction validation."""

    def _make_sim(self, **overrides):
        kwargs: dict[str, Any] = cast(
            dict[str, Any],
            {
                "model_cls": _DummyModel,
                "train_set": _dummy_dataset(),
                "test_set": _dummy_dataset(),
                "aggregator": Average,
                "n": 4,
                "f": 0,
                "rounds": 3,
                "batch_size": 8,
                "lr": 0.1,
            },
        )
        kwargs.update(overrides)
        return CentralisedSimulation(**kwargs)

    def test_construction_happy(self) -> None:
        """Construction with default parameters should succeed."""
        sim = self._make_sim()
        self.assertEqual(sim.n, 4)
        self.assertEqual(sim.f, 0)

    def test_invalid_lr_schedule_raises(self) -> None:
        """Invalid ``lr_schedule`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(lr_schedule="invalid")

    def test_robbins_monro_without_r_eta_raises(self) -> None:
        """Robbins-Monro schedule without ``r_eta`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(lr_schedule="robbins_monro", r_eta=None)

    def test_exponential_without_lr_decay_raises(self) -> None:
        """Exponential schedule without ``lr_decay`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(lr_schedule="exponential", lr_decay=None)

    def test_negative_r_eta_raises(self) -> None:
        """Negative ``r_eta`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(r_eta=-1)

    def test_zero_r_eta_raises(self) -> None:
        """Zero ``r_eta`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(r_eta=0)

    def test_negative_weight_decay_raises(self) -> None:
        """Negative ``weight_decay`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(weight_decay=-0.1)

    def test_negative_stop_attack_at_raises(self) -> None:
        """Negative ``stop_attack_at`` should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._make_sim(stop_attack_at=-1)


class CentralisedSimulationLifecycleTest(unittest.TestCase):
    """Test the simulation lifecycle: setup, step."""

    def setUp(self) -> None:
        """Set up a simulation instance for lifecycle testing."""
        self.sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
        )

    def test_model_raises_before_setup(self) -> None:
        """Accessing ``.model`` before ``.setup()`` should raise ``RuntimeError``."""
        with self.assertRaises(RuntimeError):
            _ = self.sim.model

    def test_setup_initialises_model(self) -> None:
        """:meth:`setup` should initialise the model."""
        self.sim.setup()
        assert self.sim._model is not None
        self.assertIsNotNone(self.sim._model.module)

    def test_setup_creates_worker_loaders(self) -> None:
        """:meth:`setup` should create one data loader per worker."""
        self.sim.setup()
        self.assertEqual(len(self.sim._worker_loaders), 4)

    def test_setup_is_idempotent(self) -> None:
        """Calling :meth:`setup` twice should be safe."""
        self.sim.setup()
        self.sim.setup()
        self.assertIsNotNone(self.sim._model)

    def test_step_raises_before_setup(self) -> None:
        """Calling ``.step()`` before ``.setup()`` should raise ``RuntimeError``."""
        with self.assertRaises(RuntimeError):
            self.sim.step()

    def test_step_updates_round_counter(self) -> None:
        """Each :meth:`step` should increment the round counter."""
        self.sim.setup()
        self.assertEqual(self.sim._current_round, 0)
        self.sim.step()
        self.assertEqual(self.sim._current_round, 1)


class CentralisedSimulationStepTest(unittest.TestCase):
    """Test step behaviour with honest workers and attacks."""

    def test_step_no_attack(self) -> None:
        """Step without attack should update model parameters."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
        )
        sim.setup()
        assert sim._model is not None
        params_before = sim._model.parameters.clone()
        sim.step()
        params_after = sim._model.parameters
        self.assertFalse(torch.equal(params_before, params_after))

    def test_step_with_attack(self) -> None:
        """Step with :class:`GaussianAttack` should complete successfully."""
        from krum.primitives.attacks.gaussian import GaussianAttack

        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            attack=GaussianAttack,
            attack_kwargs={"std": 1.0},
            n=4,
            f=1,
            rounds=5,
            batch_size=8,
            lr=0.1,
        )
        sim.setup()
        sim.step()
        self.assertEqual(sim._current_round, 1)

    def test_step_stop_attack_at(self) -> None:
        """``stop_attack_at=0`` should disable the attack after round 0."""
        from krum.primitives.attacks.gaussian import GaussianAttack

        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            attack=GaussianAttack,
            attack_kwargs={"std": 1.0},
            n=4,
            f=1,
            rounds=5,
            batch_size=8,
            lr=0.1,
            stop_attack_at=0,
        )
        sim.setup()
        sim.step()
        self.assertEqual(sim._current_round, 1)

    def test_worker_gradients_are_distinct(self) -> None:
        """Honest workers on different data shards must produce distinct gradients.

        This is a regression test for a bug where ``module.zero_grad()`` (with
        PyTorch's default ``set_to_none=True``) discarded the relinked flat
        gradient view, causing every worker to return the stale gradient from
        the first worker.
        """
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=128),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
        )
        sim.setup()
        gradients = [sim._train_one_worker(loader) for loader in sim._worker_loaders]
        for i in range(1, len(gradients)):
            self.assertFalse(
                torch.equal(gradients[0], gradients[i]),
                "Two honest workers produced identical gradients; the flat gradient view is likely stale.",
            )


class CentralisedSimulationLRScheduleTest(unittest.TestCase):
    """Test learning-rate schedules."""

    def test_lr_none_schedule(self) -> None:
        """The ``"none"`` schedule should keep the learning rate constant."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
            lr_schedule="none",
        )
        sim.setup()
        sim.step()
        self.assertEqual(sim._current_lr, 0.1)

    def test_lr_exponential_schedule(self) -> None:
        """The ``"exponential"`` schedule should decay the learning rate."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
            lr_schedule="exponential",
            lr_decay=0.5,
        )
        sim.setup()
        sim.step()
        self.assertAlmostEqual(sim._current_lr, 0.05)

    def test_lr_robbins_monro_schedule(self) -> None:
        """The ``"robbins_monro"`` schedule should follow ``r_eta * lr / (t + 1)``."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
            lr_schedule="robbins_monro",
            r_eta=1.0,
        )
        sim.setup()
        sim.step()
        expected = 1.0 * 0.1 / (0 + 1.0)
        self.assertAlmostEqual(sim._current_lr, expected)


class CentralisedSimulationWeightDecayTest(unittest.TestCase):
    """Test weight-decay and Xavier init behaviour."""

    def test_step_with_weight_decay(self) -> None:
        """Step with ``weight_decay`` should complete successfully."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
            weight_decay=1e-4,
        )
        sim.setup()
        sim.step()
        self.assertEqual(sim._current_round, 1)

    def test_xavier_init_zeroes_biases(self) -> None:
        """Xavier initialisation should zero the biases."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
            xavier_init=True,
        )
        sim.setup()
        assert sim._model is not None
        module = sim._model.module
        assert isinstance(module, _DummyModel)
        bias = module.linear.bias
        self.assertTrue(torch.all(bias == 0.0).item())

    def test_xavier_init_is_deterministic_across_setup(self) -> None:
        """Two ``setup()`` calls with the same seed must yield identical Xavier weights.

        Regression guard for the deterministic-Xavier contract: the local
        generator used by ``_xavier_init_`` must not be perturbed by RNG
        consumption elsewhere in the process.
        """
        kwargs: dict[str, Any] = {
            "model_cls": _DummyModel,
            "train_set": _dummy_dataset(n=32),
            "test_set": _dummy_dataset(),
            "aggregator": Average,
            "n": 4,
            "f": 0,
            "rounds": 1,
            "batch_size": 8,
            "lr": 0.1,
            "xavier_init": True,
            "seed": 42,
        }
        sim_a = CentralisedSimulation(**kwargs)
        sim_a.setup()
        sim_b = CentralisedSimulation(**kwargs)
        sim_b.setup()
        assert sim_a._model is not None and sim_b._model is not None
        torch.testing.assert_close(sim_a._model.parameters, sim_b._model.parameters)

    def test_xavier_init_ignores_global_rng(self) -> None:
        """Consuming the global RNG before ``setup()`` must not perturb Xavier weights."""
        kwargs: dict[str, Any] = {
            "model_cls": _DummyModel,
            "train_set": _dummy_dataset(n=32),
            "test_set": _dummy_dataset(),
            "aggregator": Average,
            "n": 4,
            "f": 0,
            "rounds": 1,
            "batch_size": 8,
            "lr": 0.1,
            "xavier_init": True,
            "seed": 42,
        }
        sim_ref = CentralisedSimulation(**kwargs)
        sim_ref.setup()
        assert sim_ref._model is not None
        ref_params = sim_ref._model.parameters.clone()

        # Drain the global RNG between the two setups.
        torch.manual_seed(0)
        _ = torch.randn(1000)
        _ = torch.randn(2000)

        sim_after = CentralisedSimulation(**kwargs)
        sim_after.setup()
        assert sim_after._model is not None
        torch.testing.assert_close(ref_params, sim_after._model.parameters)


class CentralisedSimulationAggregatorOverrideTest(unittest.TestCase):
    """Test that ``aggregator_kwargs`` can override the simulation's ``self.f``."""

    def test_aggregator_kwargs_f_overrides_simulation_f(self) -> None:
        """``aggregator_kwargs['f']`` must take precedence over ``self.f``."""
        captured: dict[str, Any] = {}

        class _SpyAggregator:
            @classmethod
            def aggregate(cls, gradients, /, **kwargs):
                captured.update(kwargs)
                return gradients.mean(dim=0)

        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=_SpyAggregator,  # ty:ignore[invalid-argument-type]
            aggregator_kwargs={"f": 5},
            n=10,
            f=2,
            rounds=1,
            batch_size=8,
            lr=0.1,
        )
        sim.setup()
        sim.step()
        self.assertEqual(captured["f"], 5)
        self.assertEqual(captured["n"], 10)

    def test_aggregator_kwargs_n_overrides_simulation_n(self) -> None:
        """``aggregator_kwargs['n']`` must take precedence over ``self.n``."""
        captured: dict[str, Any] = {}

        class _SpyAggregator:
            @classmethod
            def aggregate(cls, gradients, /, **kwargs):
                captured.update(kwargs)
                return gradients.mean(dim=0)

        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(),
            aggregator=_SpyAggregator,  # ty:ignore[invalid-argument-type]
            aggregator_kwargs={"n": 99},
            n=10,
            f=2,
            rounds=1,
            batch_size=8,
            lr=0.1,
        )
        sim.setup()
        sim.step()
        self.assertEqual(captured["n"], 99)
        self.assertEqual(captured["f"], 2)


if __name__ == "__main__":
    unittest.main()
