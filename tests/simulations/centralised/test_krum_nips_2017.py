"""Tests for KrumSimulation."""

import unittest
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from krum.primitives.aggregators.average import Average
from krum.simulations.centralised import KrumSimulation


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


class KrumSimulationConstructionTest(unittest.TestCase):
    """Construction and default values."""

    def _make_sim(self, **overrides):
        """Build a simulation with test defaults."""
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
        return KrumSimulation(**kwargs)

    def test_construction_defaults(self) -> None:
        """Construction with default parameters should apply NIPS 2017 values."""
        sim = self._make_sim()
        self.assertEqual(sim.lr_schedule, "exponential")

    def test_construction_happy(self) -> None:
        """Construction with default parameters should succeed."""
        sim = self._make_sim()
        self.assertEqual(sim.n, 4)
        self.assertEqual(sim.f, 0)


class KrumSimulationLifecycleTest(unittest.TestCase):
    """Lifecycle: setup, step, evaluate, run."""

    def setUp(self) -> None:
        """Set up a simulation instance for lifecycle testing."""
        self.sim = KrumSimulation(
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

    def test_setup_initialises_model(self) -> None:
        """:meth:`setup` should initialise the model."""
        self.sim.setup()
        assert self.sim._model is not None
        self.assertIsNotNone(self.sim._model.module)

    def test_evaluate_returns_error_and_loss(self) -> None:
        """:meth:`evaluate` should return a 2-tuple of floats."""
        self.sim.setup()
        result = self.sim.evaluate()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for val in result:
            self.assertIsInstance(val, float)

    def test_step_and_evaluate_workflow(self) -> None:
        """``step`` then ``evaluate`` should return metrics across multiple rounds."""
        self.sim.setup()
        for _ in range(3):
            self.sim.step()
        result = self.sim.evaluate()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class KrumSimulationLRScheduleTest(unittest.TestCase):
    """KrumSimulation uses inherited LR defaults."""

    def test_default_schedule_is_exponential(self) -> None:
        """Default LR schedule should be exponential with decay 0.99."""
        sim = KrumSimulation(
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
        self.assertEqual(sim.lr_schedule, "exponential")
        self.assertAlmostEqual(sim.lr_decay, 0.99)  # ty:ignore[no-matching-overload]


if __name__ == "__main__":
    unittest.main()
