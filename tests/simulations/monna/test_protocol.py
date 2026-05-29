"""Tests for the MoNNA simulation protocol."""

import unittest

import torch
from torch import nn

from krum.primitives import Model
from krum.primitives.aggregators import NearestNeighbor
from krum.primitives.attacks import SignFlipAttack
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
        seed: int | None = None,
    ) -> MonnaSimulation:
        """Create a tiny simulation for method-level tests."""
        module = nn.Linear(1, 1, bias=False)
        data = [[(torch.tensor([[1.0]]), torch.tensor([[1.0]]))] for _ in range(num_honest)]
        return MonnaSimulation(
            model=Model(module),
            data=data,
            loss_fn=nn.MSELoss(),
            aggregator=NearestNeighbor(n=num_honest, f=num_byzantine),
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            learning_rate=learning_rate,
            beta=beta,
            attack=attack,
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
            aggregator=NearestNeighbor(n=3, f=0),
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
            attack=SignFlipAttack(),
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
            aggregator=NearestNeighbor(n=2, f=0),
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
                aggregator=NearestNeighbor(n=2, f=1),
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
            aggregator=NearestNeighbor(n=2, f=1),
            num_honest=2,
            num_byzantine=1,
            learning_rate=0.1,
            beta=0.0,
            attack=SignFlipAttack(),
        )

        result = simulation.step()

        self.assertEqual(result["byzantine_parameters"].shape, (1, 1))
        self.assertEqual(result["mixed_parameters"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
