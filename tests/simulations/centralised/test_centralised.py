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


def _dummy_evaluator(sim: CentralisedSimulation) -> float:
    """Eval stub that returns a constant."""
    return 0.0


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
                "evaluate_fn": _dummy_evaluator,
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

    def test_missing_evaluate_fn_raises(self) -> None:
        """Missing ``evaluate_fn`` should raise ``TypeError``."""
        with self.assertRaises(TypeError):
            CentralisedSimulation(
                model_cls=_DummyModel,
                train_set=_dummy_dataset(),
                test_set=_dummy_dataset(),
                aggregator=Average,
                n=4,
                f=0,
                rounds=3,
                batch_size=8,
                lr=0.1,
            )


class CentralisedSimulationLifecycleTest(unittest.TestCase):
    """Test the simulation lifecycle: setup → step → evaluate → run."""

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
            evaluate_fn=_dummy_evaluator,
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

    def test_evaluate_delegates_to_fn(self) -> None:
        """:meth:`evaluate` should delegate to ``evaluate_fn``."""
        self.sim.setup()
        result = self.sim.evaluate()
        self.assertEqual(result, 0.0)

    def test_evaluate_test_error_raises_before_setup(self) -> None:
        """:meth:`evaluate_test_error` before setup should raise ``RuntimeError``."""
        with self.assertRaises(RuntimeError):
            self.sim.evaluate_test_error()

    def test_evaluate_test_error_and_loss_raises_before_setup(self) -> None:
        """:meth:`evaluate_test_error_and_loss` before setup should raise ``RuntimeError``."""
        with self.assertRaises(RuntimeError):
            self.sim.evaluate_test_error_and_loss()

    def test_evaluate_full_raises_before_setup(self) -> None:
        """:meth:`evaluate_full` before setup should raise ``RuntimeError``."""
        with self.assertRaises(RuntimeError):
            self.sim.evaluate_full()

    def test_run_returns_correct_trace_count(self) -> None:
        """:meth:`run` should return the expected number of traces."""
        traces = self.sim.run()
        self.assertEqual(len(traces), 2)

    def test_run_raises_on_double_call(self) -> None:
        """Calling :meth:`run` twice should raise ``RuntimeError``."""
        self.sim.run()
        with self.assertRaises(RuntimeError):
            self.sim.run()


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
            evaluate_fn=_dummy_evaluator,
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
            evaluate_fn=_dummy_evaluator,
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
            evaluate_fn=_dummy_evaluator,
        )
        sim.setup()
        sim.step()
        self.assertEqual(sim._current_round, 1)


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
            evaluate_fn=_dummy_evaluator,
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
            evaluate_fn=_dummy_evaluator,
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
            evaluate_fn=_dummy_evaluator,
        )
        sim.setup()
        sim.step()
        expected = 1.0 * 0.1 / (0 + 1.0)
        self.assertAlmostEqual(sim._current_lr, expected)


class CentralisedSimulationEvaluateHelpersTest(unittest.TestCase):
    """Test the built-in evaluate helper methods on real data."""

    def setUp(self) -> None:
        """Set up a simulation instance for evaluate helper testing."""
        self.sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(n=16),
            aggregator=Average,
            n=4,
            f=0,
            rounds=5,
            batch_size=8,
            lr=0.1,
            evaluate_fn=_dummy_evaluator,
        )
        self.sim.setup()

    def test_evaluate_test_error_returns_float(self) -> None:
        """:meth:`evaluate_test_error` should return a float in ``[0, 1]``."""
        err = self.sim.evaluate_test_error()
        self.assertIsInstance(err, float)
        self.assertGreaterEqual(err, 0.0)
        self.assertLessEqual(err, 1.0)

    def test_evaluate_test_error_and_loss_returns_dict(self) -> None:
        """:meth:`evaluate_test_error_and_loss` should return a dict."""
        result = self.sim.evaluate_test_error_and_loss()
        self.assertIsInstance(result, dict)
        self.assertSetEqual(set(result.keys()), {"test_error", "test_loss"})
        self.assertIsInstance(result["test_error"], float)
        self.assertIsInstance(result["test_loss"], float)
        self.assertGreaterEqual(result["test_error"], 0.0)
        self.assertLessEqual(result["test_error"], 1.0)

    def test_evaluate_full_returns_dict(self) -> None:
        """:meth:`evaluate_full` should return a dict."""
        result = self.sim.evaluate_full()
        self.assertIsInstance(result, dict)
        self.assertSetEqual(set(result.keys()), {"train_loss", "test_accuracy", "test_loss"})
        for v in result.values():
            self.assertIsInstance(v, float)
        self.assertGreaterEqual(result["test_accuracy"], 0.0)
        self.assertLessEqual(result["test_accuracy"], 1.0)


class CentralisedSimulationCompositionTest(unittest.TestCase):
    """Test that evaluate_fn composition works for custom evaluators."""

    def test_custom_evaluator_receives_sim(self) -> None:
        """A custom ``evaluate_fn`` should receive the simulation instance."""
        received: list[CentralisedSimulation | None] = [None]

        def custom_eval(sim: CentralisedSimulation) -> str:
            received[0] = sim
            return "custom"

        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(),
            test_set=_dummy_dataset(),
            aggregator=Average,
            n=4,
            f=0,
            rounds=3,
            batch_size=8,
            lr=0.1,
            evaluate_fn=custom_eval,
        )
        sim.setup()
        result = sim.evaluate()
        self.assertEqual(result, "custom")
        self.assertIs(received[0], sim)

    def test_use_built_in_evaluate_test_error(self) -> None:
        """:meth:`evaluate` should work when ``evaluate_fn`` is :meth:`evaluate_test_error`."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(n=16),
            aggregator=Average,
            n=4,
            f=0,
            rounds=3,
            batch_size=8,
            lr=0.1,
            evaluate_fn=CentralisedSimulation.evaluate_test_error,
        )
        sim.setup()
        result = sim.evaluate()
        self.assertIsInstance(result, float)

    def test_use_built_in_evaluate_full(self) -> None:
        """:meth:`evaluate` should work when ``evaluate_fn`` is :meth:`evaluate_full`."""
        sim = CentralisedSimulation(
            model_cls=_DummyModel,
            train_set=_dummy_dataset(n=64),
            test_set=_dummy_dataset(n=16),
            aggregator=Average,
            n=4,
            f=0,
            rounds=3,
            batch_size=8,
            lr=0.1,
            evaluate_fn=CentralisedSimulation.evaluate_full,
        )
        sim.setup()
        result = sim.evaluate()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)


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
            evaluate_fn=_dummy_evaluator,
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
            evaluate_fn=_dummy_evaluator,
        )
        sim.setup()
        assert sim._model is not None
        module = sim._model.module
        assert isinstance(module, _DummyModel)
        bias = module.linear.bias
        self.assertTrue(torch.all(bias == 0.0).item())


if __name__ == "__main__":
    unittest.main()
