"""Tests for the MoNNA simulation protocol."""

import unittest
from functools import partial

import torch
from torch import nn

from krum.primitives import Model
from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.simulations.monna import MonnaSimulation


class MonnaProtocolTest(unittest.TestCase):
    """Test MoNNA simulation behavior."""

    def make_simulation(
        self,
        *,
        num_honest: int,
        num_byzantine: int,
        learning_rate: float,
        beta: float = 0.99,
        attack=None,
        byzantine_reach: str = "all",
        seed: int | None = None,
    ) -> MonnaSimulation:
        """Create a tiny simulation for method-level tests."""
        module = nn.Linear(1, 1, bias=False)
        data = [[(torch.tensor([[1.0]]), torch.tensor([[1.0]]))] for _ in range(num_honest)]
        return MonnaSimulation(
            model=Model(module),
            data=data,
            loss_fn=nn.MSELoss(),
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            learning_rate=learning_rate,
            beta=beta,
            attack=attack,
            byzantine_reach=byzantine_reach,
            seed=seed,
        )

    def test_simulation_initializes_worker_parameters_from_model(self) -> None:
        """The simulation owns initial parameters and momentum directly."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(2.0)
        data = [
            [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))],
            [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))],
            [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))],
        ]

        simulation = MonnaSimulation(
            model=Model(module),
            data=data,
            loss_fn=nn.MSELoss(),
            num_honest=3,
            num_byzantine=0,
            learning_rate=0.1,
        )

        self.assertEqual(simulation.parameters.shape, (3, 1))
        self.assertTrue(torch.equal(simulation.parameters, torch.tensor([[2.0], [2.0], [2.0]])))
        self.assertTrue(torch.equal(simulation.momentum, torch.zeros((3, 1))))
        self.assertEqual(simulation.step_index, 0)

    def test_update_local_momentum_returns_worker_side_formula(self) -> None:
        """Momentum is updated independently for each worker."""
        simulation = self.make_simulation(num_honest=2, num_byzantine=0, learning_rate=0.1, beta=0.5)
        simulation.momentum = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gradients = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        result = simulation.update_local_momentum(gradients)

        expected = torch.tensor([[5.5, 11.0], [16.5, 22.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_aggregate_over_received_nodes_averages_n_minus_2f_vectors(self) -> None:
        """Each worker runs NNA on n - f local vectors and averages n - 2f of them."""
        honest = torch.tensor([[0.0], [10.0], [20.0]])
        byzantine = torch.tensor([[100.0]])
        simulation = self.make_simulation(
            num_honest=3,
            num_byzantine=1,
            learning_rate=0.1,
            attack=SignFlipAttack.generate,
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
        data = [
            [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))],
            [(torch.tensor([[2.0]]), torch.tensor([[2.0]]))],
        ]
        simulation = MonnaSimulation(
            model=model,
            data=data,
            loss_fn=nn.MSELoss(),
            num_honest=2,
            num_byzantine=0,
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

    def test_defaults_to_nearest_neighbor_average_sized_to_n_minus_2f(self) -> None:
        """MoNNA owns the mixing rule: default NNA keeps num_honest - num_byzantine."""
        simulation = self.make_simulation(
            num_honest=5, num_byzantine=2, learning_rate=0.1, attack=SignFlipAttack.generate
        )

        self.assertIsInstance(simulation.aggregator, partial)
        self.assertIs(simulation.aggregator.func.__self__, NearestNeighborAverage)
        self.assertEqual(simulation.aggregator.keywords["num_closest"], 3)

    def test_accepts_aggregator_override(self) -> None:
        """A supplied aggregator replaces the default mixing rule."""
        module = nn.Linear(1, 1, bias=False)
        data = [[(torch.tensor([[1.0]]), torch.tensor([[1.0]]))] for _ in range(3)]
        custom = partial(NearestNeighborAverage.aggregate, num_closest=1)

        simulation = MonnaSimulation(
            model=Model(module),
            data=data,
            loss_fn=nn.MSELoss(),
            num_honest=3,
            num_byzantine=0,
            learning_rate=0.1,
            aggregator=custom,
        )

        self.assertIs(simulation.aggregator, custom)

    def test_simulation_requires_attack_when_byzantine_workers_are_configured(self) -> None:
        """Byzantine rounds need an explicit attack implementation."""
        module = nn.Linear(1, 1, bias=False)
        data = [
            [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))],
            [(torch.tensor([[2.0]]), torch.tensor([[2.0]]))],
        ]

        with self.assertRaises(ValueError):
            MonnaSimulation(
                model=Model(module),
                data=data,
                loss_fn=nn.MSELoss(),
                num_honest=2,
                num_byzantine=1,
                learning_rate=0.1,
            )

    def test_simulation_accepts_attack_for_byzantine_parameters(self) -> None:
        """Attack-generated parameter vectors participate in mixing."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(0.0)
        data = [
            [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))],
            [(torch.tensor([[2.0]]), torch.tensor([[2.0]]))],
        ]
        simulation = MonnaSimulation(
            model=Model(module),
            data=data,
            loss_fn=nn.MSELoss(),
            num_honest=2,
            num_byzantine=1,
            learning_rate=0.1,
            beta=0.0,
            attack=SignFlipAttack.generate,
        )

        result = simulation.step()

        self.assertEqual(result["byzantine_parameters"].shape, (1, 1))
        self.assertEqual(result["mixed_parameters"].shape, (2, 1))

    def test_byzantine_reach_rejects_unknown_value(self) -> None:
        """Only the two documented reach modes are accepted."""
        with self.assertRaises(ValueError):
            self.make_simulation(
                num_honest=3,
                num_byzantine=1,
                learning_rate=0.1,
                attack=SignFlipAttack.generate,
                byzantine_reach="everyone",
            )

    def test_gathered_set_keeps_n_minus_f_size_with_self_first_in_both_modes(self) -> None:
        """Every received set holds n - f models led by the worker's own model."""
        # n = 7: 5 honest, 2 Byzantine; each worker receives n - f = 5 models.
        honest = torch.arange(5.0).unsqueeze(1)  # ids 0..4
        byzantine = torch.tensor([[100.0], [101.0]])  # ids 100, 101

        for reach in ("all", "sampled"):
            simulation = self.make_simulation(
                num_honest=5,
                num_byzantine=2,
                learning_rate=0.1,
                attack=SignFlipAttack.generate,
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
            num_honest=5,
            num_byzantine=2,
            learning_rate=0.1,
            attack=SignFlipAttack.generate,
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
            num_honest=5,
            num_byzantine=2,
            learning_rate=0.1,
            attack=SignFlipAttack.generate,
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
        simulation = self.make_simulation(num_honest=4, num_byzantine=0, learning_rate=0.1, seed=0)

        received = simulation.gather_received_models(honest, byzantine, worker_index=1)

        self.assertEqual(received.shape, (4, 1))
        self.assertEqual(received[0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
