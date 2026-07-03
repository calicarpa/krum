"""Tests for DecentralisedSimulation base class."""

import unittest
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn

from krum.primitives import Model
from krum.primitives.aggregators.average import Average
from krum.simulations.decentralised import Batch, DecentralisedSimulation, LossFn, StepResult


class _ConcreteSimulation(DecentralisedSimulation[StepResult]):
    """Minimal concrete subclass for testing base class methods."""

    def local_update(self, gradients: torch.Tensor) -> torch.Tensor:
        return self.parameters - gradients

    def gather_received_models(
        self,
        honest_vectors: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        *,
        worker_index: int,
    ) -> torch.Tensor:
        return torch.cat([honest_vectors, byzantine_parameters], dim=0)

    def build_step_result(
        self,
        *,
        honest_gradients: torch.Tensor,
        local_parameters: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        mixed_parameters: torch.Tensor,
        losses: torch.Tensor,
    ) -> StepResult:
        return {
            "step": self.step_index,
            "parameters": self.parameters.detach().clone(),
            "honest_gradients": honest_gradients.detach().clone(),
            "local_parameters": local_parameters.detach().clone(),
            "byzantine_parameters": byzantine_parameters.detach().clone(),
            "mixed_parameters": mixed_parameters.detach().clone(),
            "losses": losses.detach().clone(),
        }


def _make_simulation(
    *,
    n: int,
    f: int,
    model: Model | None = None,
    data: Sequence[Iterable[Batch]] | None = None,
    loss_fn: LossFn | None = None,
    attack=None,
    seed: int | None = None,
) -> _ConcreteSimulation:
    """Factory for a tiny concrete simulation."""
    if model is None:
        model = Model(nn.Linear(1, 1, bias=False))
    if data is None:
        data = [[(torch.tensor([[1.0]]), torch.tensor([[1.0]]))] for _ in range(n - f)]
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    return _ConcreteSimulation(
        model=model,
        data=data,
        loss_fn=loss_fn,
        n=n,
        f=f,
        attack=attack,
        aggregator=Average,
        seed=seed,
    )


class DecentralisedSimulationConstructionTest(unittest.TestCase):
    """Test construction validation."""

    def test_rejects_negative_f(self) -> None:
        """Negative Byzantine count is invalid."""
        with self.assertRaises(ValueError):
            _make_simulation(n=4, f=-1)

    def test_requires_at_least_one_honest_worker(self) -> None:
        """N - f must be at least 1."""
        with self.assertRaises(ValueError):
            _make_simulation(n=4, f=4)

    def test_requires_enough_honest_workers_for_mixing(self) -> None:
        """N - f must exceed f for decentralised mixing."""
        with self.assertRaises(ValueError):
            _make_simulation(n=4, f=3)

    def test_rejects_too_few_data_streams(self) -> None:
        """Number of data streams must equal n - f."""
        data = [[(torch.tensor([[1.0]]), torch.tensor([[1.0]]))] for _ in range(2)]
        with self.assertRaises(ValueError):
            _make_simulation(n=7, f=1, data=data)

    def test_requires_attack_when_f_greater_than_zero(self) -> None:
        """An attack is required when f > 0."""
        with self.assertRaises(ValueError):
            _make_simulation(n=5, f=1)

    def test_requires_model_instance(self) -> None:
        """Model must be a Model instance."""
        with self.assertRaises(TypeError):
            _make_simulation(model="not_a_model", n=4, f=0)  # ty:ignore[invalid-argument-type]

    def test_requires_callable_loss_fn(self) -> None:
        """loss_fn must be callable."""
        with self.assertRaises(TypeError):
            _make_simulation(loss_fn="not_callable", n=4, f=0)  # ty:ignore[invalid-argument-type]

    def test_rejects_non_attack_subclass(self) -> None:
        """Attack must be an Attack subclass."""
        with self.assertRaises(TypeError):
            _make_simulation(n=5, f=1, attack=object)  # type: ignore[arg-type]

    def test_rejects_non_aggregator_subclass(self) -> None:
        """Aggregator must be an Aggregator subclass."""
        with self.assertRaises(TypeError):
            _make_simulation(n=4, f=0, aggregator=object)  # ty:ignore[unknown-argument]

    def test_accepts_seed_as_int(self) -> None:
        """An integer seed is accepted."""
        sim = _make_simulation(n=4, f=0, seed=42)
        self.assertIsNotNone(sim.generator)

    def test_rejects_non_int_seed(self) -> None:
        """A non-integer seed raises TypeError."""
        with self.assertRaises(TypeError):
            _make_simulation(n=4, f=0, seed=1.5)  # ty:ignore[invalid-argument-type]

    def test_stores_construction_parameters(self) -> None:
        """All init parameters are accessible as attributes."""
        sim = _make_simulation(n=5, f=0)
        self.assertEqual(sim.n, 5)
        self.assertEqual(sim.f, 0)
        self.assertEqual(sim.num_honest, 5)
        self.assertIs(sim.aggregator, Average)

    def test_initializes_parameters_from_model(self) -> None:
        """Parameters is a stacked clone of the model flat params."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(3.0)
        sim = _make_simulation(model=Model(module), n=4, f=0)
        expected = torch.full((4, 1), 3.0)
        self.assertTrue(torch.equal(sim.parameters, expected))
        self.assertEqual(sim.step_index, 0)

    def test_initializes_with_zero_byzantine_parameters_when_f_is_zero(self) -> None:
        """generate_byzantine_models returns empty tensor when f = 0."""
        sim = _make_simulation(n=4, f=0)
        result = sim.generate_byzantine_models(sim.parameters)
        self.assertEqual(result.shape, (0, 1))


class DecentralisedSimulationMethodTest(unittest.TestCase):
    """Test base class concrete methods."""

    def make_simulation(self, **overrides) -> _ConcreteSimulation:
        """Return a tiny concrete simulation with optional overrides."""
        kwargs = {"n": 4, "f": 0}
        kwargs.update(overrides)
        return _make_simulation(**kwargs)  # ty:ignore[invalid-argument-type]

    def test_collect_worker_batches_returns_one_batch_per_worker(self) -> None:
        """Each honest worker contributes one batch per collect."""
        sim = self.make_simulation(n=3)
        batches = sim.collect_worker_batches()
        self.assertEqual(len(batches), 3)
        for inputs, targets in batches:
            self.assertIsInstance(inputs, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)

    def test_compute_honest_worker_gradients_produces_stacked_tensors(self) -> None:
        """Gradients and losses are stacked with one row per worker."""
        sim = self.make_simulation(n=2)
        batches = sim.collect_worker_batches()
        gradients, losses = sim.compute_honest_worker_gradients(batches)
        self.assertEqual(gradients.shape, (2, 1))
        self.assertEqual(losses.shape, (2,))

    def test_copy_parameters_to_model_loads_flat_vector(self) -> None:
        """Copying parameters into the model makes them accessible."""
        module = nn.Linear(1, 1, bias=False)
        model = Model(module)
        sim = self.make_simulation(model=model, n=2)
        new_params = torch.tensor([[5.0], [5.0]])
        sim.parameters = new_params
        sim.copy_parameters_to_model(new_params[0])
        self.assertTrue(torch.equal(model.parameters, new_params[0]))

    def test_generate_byzantine_models_returns_empty_when_no_attack(self) -> None:
        """With f = 0, the Byzantine set is always empty."""
        sim = self.make_simulation(n=4, f=0)
        local = sim.parameters
        result = sim.generate_byzantine_models(local)
        self.assertEqual(result.shape, (0, 1))

    def test_aggregate_received_models_forwards_to_aggregator(self) -> None:
        """aggregate_received_models delegates to the configured aggregator."""
        sim = self.make_simulation(n=3, f=0)
        candidates = torch.tensor([[1.0], [2.0], [3.0]])
        pivot = torch.tensor([1.0])
        result = sim.aggregate_received_models(candidates, pivot=pivot)
        # Average of [1, 2, 3] = 2
        self.assertAlmostEqual(result.item(), 2.0)

    def test_commit_state_updates_step_index_and_parameters(self) -> None:
        """State commit advances the counter and stores new parameters."""
        sim = self.make_simulation(n=4, f=0)
        new_params = torch.full((4, 1), 99.0)
        sim.commit_state(new_params)
        self.assertEqual(sim.step_index, 1)
        self.assertTrue(torch.equal(sim.parameters, new_params))

    def test_commit_state_clones_parameters(self) -> None:
        """Committed parameters are detached clones."""
        sim = self.make_simulation(n=4, f=0)
        new_params = torch.full((4, 1), 99.0)
        sim.commit_state(new_params)
        new_params.fill_(0.0)
        self.assertFalse(torch.equal(sim.parameters, new_params))


class DecentralisedSimulationRunTest(unittest.TestCase):
    """Test run driver."""

    def test_run_with_zero_rounds_returns_empty_list(self) -> None:
        """run(0) is a no-op."""
        sim = _make_simulation(n=2, f=0)
        results = sim.run(0)
        self.assertEqual(results, [])
        self.assertEqual(sim.step_index, 0)

    def test_run_rejects_negative_rounds(self) -> None:
        """Negative rounds raise ValueError."""
        sim = _make_simulation(n=2, f=0)
        with self.assertRaises(ValueError):
            sim.run(-1)

    def test_run_executes_steps_and_collects_results(self) -> None:
        """Each round produces one result with incremented step."""
        batch = (torch.tensor([[1.0]]), torch.tensor([[1.0]]))
        data = [[batch, batch] for _ in range(2)]
        sim = _make_simulation(n=2, f=0, data=data)
        results = sim.run(2)
        self.assertEqual(len(results), 2)
        self.assertEqual([r["step"] for r in results], [1, 2])


if __name__ == "__main__":
    unittest.main()
