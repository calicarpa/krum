"""Tests for the MoNNA simulation protocol."""

import unittest

import torch
from torch import nn

from krum.primitives import Model
from krum.primitives.attacks import SignFlipAttack
from krum.simulations.monna import MonnaConfig, compute_momentum, initial_state, mix_each_worker, run_round


class MonnaProtocolTest(unittest.TestCase):
    """Test MoNNA protocol helpers."""

    def test_initial_state_copies_model_parameters_to_each_worker(self) -> None:
        """Initial state gives every honest worker the same starting parameter vector."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(2.0)

        state = initial_state(Model(module), num_honest=3)

        self.assertEqual(state.parameters.shape, (3, 1))
        self.assertTrue(torch.equal(state.parameters, torch.tensor([[2.0], [2.0], [2.0]])))
        self.assertTrue(torch.equal(state.momentum, torch.zeros((3, 1))))

    def test_compute_momentum_returns_worker_side_formula(self) -> None:
        """Momentum is updated independently for each worker."""
        previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gradients = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        result = compute_momentum(previous, gradients, beta=0.5)

        expected = torch.tensor([[5.5, 11.0], [16.5, 22.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_mix_each_worker_averages_n_minus_2f_coordination_vectors(self) -> None:
        """Each worker runs NNA on n - f local vectors and averages n - 2f of them."""
        honest = torch.tensor([[0.0], [10.0], [20.0]])
        byzantine = torch.tensor([[100.0]])
        generator = torch.Generator().manual_seed(0)

        mixed = mix_each_worker(honest, byzantine, f=1, generator=generator)

        expected = torch.tensor([[5.0], [15.0], [15.0]])
        self.assertTrue(torch.equal(mixed, expected))

    def test_mix_each_worker_requires_one_byzantine_vector_per_fault(self) -> None:
        """The coordination candidate set must contain exactly f Byzantine vectors."""
        honest = torch.tensor([[0.0], [10.0], [20.0]])
        byzantine = torch.tensor([[100.0], [200.0]])

        with self.assertRaises(ValueError):
            mix_each_worker(honest, byzantine, f=1)

    def test_run_round_computes_real_gradients_and_updates_each_worker(self) -> None:
        """One round performs real backward passes and local parameter updates."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(0.0)
        model = Model(module)
        state = initial_state(model, num_honest=2)
        config = MonnaConfig(num_honest=2, num_byzantine=0, learning_rate=0.1, beta=0.0)
        batches = [
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[2.0]]), torch.tensor([[2.0]])),
        ]
        loss_fn = nn.MSELoss()

        result = run_round(state, config=config, model=model, batches=batches, loss_fn=loss_fn)

        expected_gradients = torch.tensor([[-2.0], [-8.0]])
        expected_mixed = torch.tensor([[-5.0], [-5.0]])
        expected_parameters = torch.tensor([[0.5], [0.5]])
        self.assertEqual(result.state.step, 1)
        self.assertTrue(torch.allclose(result.honest_gradients, expected_gradients))
        self.assertTrue(torch.allclose(result.mixed_vectors, expected_mixed))
        self.assertTrue(torch.allclose(result.state.parameters, expected_parameters))

    def test_run_round_requires_attack_when_byzantine_workers_are_configured(self) -> None:
        """Byzantine rounds need an explicit attack implementation."""
        module = nn.Linear(1, 1, bias=False)
        model = Model(module)
        state = initial_state(model, num_honest=2)
        config = MonnaConfig(num_honest=2, num_byzantine=1, learning_rate=0.1)
        batches = [
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[2.0]]), torch.tensor([[2.0]])),
        ]

        with self.assertRaises(ValueError):
            run_round(state, config=config, model=model, batches=batches, loss_fn=nn.MSELoss())

    def test_run_round_accepts_attack_for_byzantine_vectors(self) -> None:
        """Attack-generated vectors participate in nearest-neighbor mixing."""
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(0.0)
        model = Model(module)
        state = initial_state(model, num_honest=2)
        config = MonnaConfig(num_honest=2, num_byzantine=1, learning_rate=0.1, beta=0.0)
        batches = [
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[2.0]]), torch.tensor([[2.0]])),
        ]

        result = run_round(
            state,
            config=config,
            model=model,
            batches=batches,
            loss_fn=nn.MSELoss(),
            attack=SignFlipAttack(),
        )

        self.assertEqual(result.byzantine_vectors.shape, (1, 1))
        self.assertEqual(result.mixed_vectors.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
