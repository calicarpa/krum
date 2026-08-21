"""Tests for the MoNNA simulation protocol."""

import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.primitives.data_partitioners.dirichlet import DirichletPartitioner
from krum.primitives.data_partitioners.iid import IidPartitioner
from krum.primitives.models import Model
from krum.simulations.decentralised.monna_icml_2023 import ByzantineReach, MonnaSimulation


def _dataset(x: torch.Tensor, y: torch.Tensor) -> TensorDataset:
    """Build a single-sample dataset."""
    return TensorDataset(x, y)


_TEST_SET = _dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]]))


class MonnaProtocolTest(unittest.TestCase):
    """Test MoNNA simulation behavior."""

    def make_simulation(
        self,
        *,
        n: int,
        f: int,
        learning_rate: float,
        beta: float = 0.99,
        weight_decay: float = 0.0,
        attack=None,
        byzantine_reach: ByzantineReach = "all",
        seed: int | None = None,
    ) -> MonnaSimulation:
        """Create a tiny simulation for method-level tests."""
        module = nn.Linear(1, 1, bias=False)
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])) for _ in range(n)]
        return MonnaSimulation(
            model=Model(module),
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=n,
            f=f,
            learning_rate=learning_rate,
            beta=beta,
            weight_decay=weight_decay,
            attack=attack,
            byzantine_reach=byzantine_reach,
            seed=seed,
        )

    def test_simulation_initializes_worker_parameters_from_model(self) -> None:
        """The simulation owns initial parameters and momentum directly."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(2.0)
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])) for _ in range(3)]

        simulation = MonnaSimulation(
            model=Model(module),
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=3,
            f=0,
            learning_rate=0.1,
        )

        self.assertEqual(simulation.parameters.shape, (3, 1))
        self.assertTrue(torch.equal(simulation.parameters, torch.tensor([[2.0], [2.0], [2.0]])))
        self.assertTrue(torch.equal(simulation.momentum, torch.zeros((3, 1))))
        self.assertEqual(simulation.step_index, 0)

    def test_update_local_momentum_returns_worker_side_formula(self) -> None:
        """Momentum is updated independently for each worker."""
        simulation = self.make_simulation(n=2, f=0, learning_rate=0.1, beta=0.5)
        simulation.momentum = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gradients = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        result = simulation.update_local_momentum(gradients)

        expected = torch.tensor([[5.5, 11.0], [16.5, 22.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_apply_weight_decay_adds_scaled_parameters_to_gradients(self) -> None:
        """Weight decay adds ``weight_decay * parameters`` to each gradient."""
        simulation = self.make_simulation(n=2, f=0, learning_rate=0.1, weight_decay=0.5)
        simulation.parameters = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gradients = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        result = simulation.apply_weight_decay(gradients)

        expected = torch.tensor([[10.5, 21.0], [31.5, 42.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_apply_weight_decay_is_noop_when_zero(self) -> None:
        """Weight decay of zero leaves gradients unchanged."""
        simulation = self.make_simulation(n=2, f=0, learning_rate=0.1, weight_decay=0.0)
        gradients = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        result = simulation.apply_weight_decay(gradients)

        self.assertTrue(torch.equal(result, gradients))

    def test_negative_weight_decay_rejected(self) -> None:
        """A negative weight_decay raises ValueError."""
        with self.assertRaises(ValueError):
            self.make_simulation(n=2, f=0, learning_rate=0.1, weight_decay=-0.1)

    def test_aggregate_over_received_nodes_averages_n_minus_2f_vectors(self) -> None:
        """Each worker runs NNA on n - f local vectors and averages n - 2f of them."""
        honest = torch.tensor([[0.0], [10.0], [20.0]])
        byzantine = torch.tensor([[100.0]])
        simulation = self.make_simulation(
            n=4,
            f=1,
            learning_rate=0.1,
            attack=SignFlipAttack,
            seed=0,
        )

        mixed = simulation.aggregate_over_received_nodes(honest, byzantine)

        expected = torch.tensor([[5.0], [15.0], [15.0]])
        self.assertTrue(torch.equal(mixed, expected))

    def test_simulation_step_shows_local_update_then_parameter_mixing(self) -> None:
        """Class API owns state and mixes post-local-update parameter vectors."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(0.0)
        model = Model(module)
        train_datasets = [
            _dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            _dataset(torch.tensor([[2.0]]), torch.tensor([[2.0]])),
        ]
        simulation = MonnaSimulation(
            model=model,
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=2,
            f=0,
            learning_rate=0.1,
            beta=0.0,
        )

        result = simulation.step()

        expected_gradients = torch.tensor([[-2.0], [-8.0]])
        expected_local_parameters = torch.tensor([[0.2], [0.8]])
        expected_mixed_parameters = torch.tensor([[0.5], [0.5]])
        self.assertEqual(result["step"], 1)
        self.assertTrue(torch.allclose(result["honest_gradients"], expected_gradients))
        self.assertTrue(torch.allclose(result["local_parameters"], expected_local_parameters))
        self.assertTrue(torch.allclose(result["mixed_parameters"], expected_mixed_parameters))
        self.assertTrue(torch.allclose(simulation.parameters, expected_mixed_parameters))

    def test_evaluate_averages_loss_and_accuracy_across_honest_workers(self) -> None:
        """evaluate() loads each worker's params in turn and averages the test metrics."""
        module = nn.Linear(1, 2, bias=False)
        test_set = TensorDataset(torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1]))
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([1])) for _ in range(2)]
        simulation = MonnaSimulation(
            model=Model(module),
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=test_set,
            test_batch_size=2,
            loss_fn=nn.CrossEntropyLoss(),
            n=2,
            f=0,
            learning_rate=0.1,
        )

        loss, accuracy = simulation.evaluate()

        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_evaluate_not_called_by_step(self) -> None:
        """step() never touches the test loader."""
        simulation = self.make_simulation(n=2, f=0, learning_rate=0.1)
        simulation.step()
        # Should not raise, and should be independently callable.
        simulation.evaluate()

    def test_defaults_to_nearest_neighbor_average_sized_to_n_minus_2f(self) -> None:
        """MoNNA owns the mixing rule: default NNA keeps n - 2f."""
        simulation = self.make_simulation(n=7, f=2, learning_rate=0.1, attack=SignFlipAttack)

        self.assertIs(simulation.aggregator, NearestNeighborAverage)
        self.assertEqual(simulation.aggregator_kwargs["num_closest"], 3)

    def test_accepts_aggregator_override(self) -> None:
        """A supplied aggregator replaces the default mixing rule."""
        module = nn.Linear(1, 1, bias=False)
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])) for _ in range(3)]

        simulation = MonnaSimulation(
            model=Model(module),
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=3,
            f=0,
            learning_rate=0.1,
            aggregator=NearestNeighborAverage,
            aggregator_kwargs={"num_closest": 1},
        )

        self.assertIs(simulation.aggregator, NearestNeighborAverage)
        self.assertEqual(simulation.aggregator_kwargs["num_closest"], 1)

    def test_simulation_requires_attack_when_byzantine_workers_are_configured(self) -> None:
        """Byzantine rounds need an explicit attack implementation."""
        module = nn.Linear(1, 1, bias=False)
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])) for _ in range(3)]

        with self.assertRaises(ValueError):
            MonnaSimulation(
                model=Model(module),
                train_datasets=train_datasets,
                train_batch_size=1,
                test_set=_TEST_SET,
                test_batch_size=1,
                loss_fn=nn.MSELoss(),
                n=3,
                f=1,
                learning_rate=0.1,
            )

    def test_simulation_accepts_attack_for_byzantine_parameters(self) -> None:
        """Attack-generated parameter vectors participate in mixing."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(0.0)
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])) for _ in range(3)]
        simulation = MonnaSimulation(
            model=Model(module),
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=3,
            f=1,
            learning_rate=0.1,
            beta=0.0,
            attack=SignFlipAttack,
        )

        result = simulation.step()

        self.assertEqual(result["byzantine_parameters"].shape, (1, 1))
        self.assertEqual(result["mixed_parameters"].shape, (2, 1))

    def test_byzantine_reach_rejects_unknown_value(self) -> None:
        """Only the two documented reach modes are accepted."""
        with self.assertRaises(ValueError):
            self.make_simulation(
                n=4,
                f=1,
                learning_rate=0.1,
                attack=SignFlipAttack,
                byzantine_reach="everyone",  # ty: ignore[invalid-argument-type]
            )

    def test_gathered_set_keeps_n_minus_f_size_with_self_first_in_both_modes(self) -> None:
        """Every received set holds n - f models led by the worker's own model."""
        # n = 7: 5 honest, 2 Byzantine; each worker receives n - f = 5 models.
        honest = torch.arange(5.0).unsqueeze(1)  # ids 0..4
        byzantine = torch.tensor([[100.0], [101.0]])  # ids 100, 101

        for reach in ("all", "sampled"):
            simulation = self.make_simulation(
                n=7,
                f=2,
                learning_rate=0.1,
                attack=SignFlipAttack,
                byzantine_reach=reach,
                seed=0,
            )
            for worker_index in range(5):
                received = simulation.gather_received_models(honest, byzantine, worker_index=worker_index)
                self.assertEqual(received.shape, (5, 1), msg=f"{reach=} {worker_index=}")
                self.assertEqual(received[0].item(), float(worker_index), msg=f"{reach=} {worker_index=}")

    def test_all_reach_injects_every_byzantine_model_into_each_worker(self) -> None:
        """``"all"`` is the worst case: all f Byzantine models reach every worker."""
        honest = torch.arange(5.0).unsqueeze(1)
        byzantine = torch.tensor([[100.0], [101.0]])
        simulation = self.make_simulation(
            n=7,
            f=2,
            learning_rate=0.1,
            attack=SignFlipAttack,
            byzantine_reach="all",
            seed=0,
        )

        for worker_index in range(5):
            received = simulation.gather_received_models(honest, byzantine, worker_index=worker_index)
            ids = set(received.squeeze(1).tolist())
            self.assertTrue({100.0, 101.0}.issubset(ids), msg=f"{worker_index=} {ids=}")

    def test_sampled_reach_draws_byzantine_models_at_random(self) -> None:
        """``"sampled"`` lets Byzantine reach vary, never exceeding f per worker."""
        honest = torch.arange(5.0).unsqueeze(1)
        byzantine = torch.tensor([[100.0], [101.0]])
        simulation = self.make_simulation(
            n=7,
            f=2,
            learning_rate=0.1,
            attack=SignFlipAttack,
            byzantine_reach="sampled",
            seed=0,
        )

        byzantine_counts = []
        for worker_index in range(5):
            received = simulation.gather_received_models(honest, byzantine, worker_index=worker_index)
            byzantine_counts.append(sum(1 for x in received.squeeze(1).tolist() if x >= 100.0))

        # Bounded by f, and not the constant f that "all" would produce.
        self.assertTrue(all(0 <= count <= 2 for count in byzantine_counts), msg=f"{byzantine_counts=}")
        self.assertNotEqual(byzantine_counts, [2, 2, 2, 2, 2])

    def test_all_reach_excludes_byzantine_when_none_configured(self) -> None:
        """With f = 0 the received set is purely honest and still sized n."""
        honest = torch.arange(4.0).unsqueeze(1)
        byzantine = honest.new_empty((0, 1))
        simulation = self.make_simulation(n=4, f=0, learning_rate=0.1, seed=0)

        received = simulation.gather_received_models(honest, byzantine, worker_index=1)

        self.assertEqual(received.shape, (4, 1))
        self.assertEqual(received[0].item(), 1.0)

    def test_run_executes_one_step_per_round_in_order(self) -> None:
        """``run`` drives ``step`` once per round and collects the snapshots."""
        module = nn.Linear(1, 1, bias=False)
        train_datasets = [_dataset(torch.tensor([[1.0]]), torch.tensor([[1.0]])) for _ in range(2)]
        simulation = MonnaSimulation(
            model=Model(module),
            train_datasets=train_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=2,
            f=0,
            learning_rate=0.1,
        )

        results = simulation.run(3)

        self.assertEqual([result["step"] for result in results], [1, 2, 3])
        self.assertEqual(simulation.step_index, 3)

    def test_run_with_zero_rounds_returns_empty_and_leaves_state(self) -> None:
        """``run(0)`` is a no-op that returns no snapshots."""
        simulation = self.make_simulation(n=2, f=0, learning_rate=0.1)

        results = simulation.run(0)

        self.assertEqual(results, [])
        self.assertEqual(simulation.step_index, 0)

    def test_run_rejects_negative_rounds(self) -> None:
        """A negative round count is a usage error."""
        simulation = self.make_simulation(n=2, f=0, learning_rate=0.1)

        with self.assertRaises(ValueError):
            simulation.run(-1)

    def test_step_with_iid_partitioner(self) -> None:
        """IidPartitioner output works as train_datasets."""
        dataset = TensorDataset(torch.randn(200, 1), torch.randint(0, 2, (200,)))
        worker_datasets = IidPartitioner.partition(dataset, n=4, seed=42)

        simulation = MonnaSimulation(
            model=Model(nn.Linear(1, 1, bias=False)),
            train_datasets=worker_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=4,
            f=0,
            learning_rate=0.1,
            beta=0.99,
            seed=42,
        )
        result = simulation.step()
        self.assertEqual(result["step"], 1)

    def test_step_with_dirichlet_partitioner(self) -> None:
        """DirichletPartitioner output works as train_datasets (non-IID)."""
        dataset = TensorDataset(torch.randn(400, 1), torch.randint(0, 2, (400,)))
        worker_datasets = DirichletPartitioner.partition(dataset, n=4, alpha=1.0, seed=42)

        # Verify no honest worker got an empty shard
        for ds in worker_datasets:
            self.assertGreater(len(ds), 0)

        simulation = MonnaSimulation(
            model=Model(nn.Linear(1, 1, bias=False)),
            train_datasets=worker_datasets,
            train_batch_size=1,
            test_set=_TEST_SET,
            test_batch_size=1,
            loss_fn=nn.MSELoss(),
            n=4,
            f=0,
            learning_rate=0.1,
            beta=0.99,
            seed=42,
        )
        result = simulation.step()
        self.assertEqual(result["step"], 1)


if __name__ == "__main__":
    unittest.main()
