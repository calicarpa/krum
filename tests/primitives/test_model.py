"""Tests for the Model class."""

import unittest

import torch
from torch import nn

from krum.primitives.model import Model


class ModelTest(unittest.TestCase):
    """Test Model zero-copy behavior."""

    def setUp(self):
        """Set up a small linear model for testing."""
        self.linear = nn.Linear(3, 2)
        self.model = Model(self.linear)

    def test_parameters_zero_copy(self) -> None:
        """Modifying the flat parameters tensor updates the model weights."""
        flat = self.model.parameters
        original = flat.clone()

        flat[0] = 99.0

        self.assertNotEqual(flat[0].item(), original[0].item())

        # Verify the first parameter was updated
        new_flat = torch.cat([p.data.view(-1) for p in self.linear.parameters()])
        self.assertEqual(new_flat[0].item(), 99.0)

    def test_parameters_update_from_module_side(self) -> None:
        """Modifying a module parameter updates the flat tensor."""
        flat = self.model.parameters
        original = flat[0].item()

        with torch.no_grad():
            self.linear.weight[0, 0] = 42.0

        self.assertEqual(flat[0].item(), 42.0)
        self.assertNotEqual(flat[0].item(), original)

    def test_parameters_cache_returns_same_object(self) -> None:
        """Accessing parameters twice returns the same tensor object."""
        flat1 = self.model.parameters
        flat2 = self.model.parameters

        self.assertIs(flat1, flat2)

    def test_parameters_cache_invalidated_on_module_change(self) -> None:
        """Setting a new module invalidates the parameters cache."""
        flat1 = self.model.parameters

        self.model.module = nn.Linear(5, 1)
        flat2 = self.model.parameters

        self.assertIsNot(flat1, flat2)

    def test_gradients_zero_copy(self) -> None:
        """Modifying the flat gradients tensor updates individual grads."""
        x = torch.randn(1, 3)
        loss = self.linear(x).sum()
        loss.backward()

        flat_grads = self.model.gradients
        flat_grads[0] = 99.0

        # Verify the first parameter's grad was updated
        new_flat = torch.cat([p.grad.data.view(-1) for p in self.linear.parameters()])
        self.assertEqual(new_flat[0].item(), 99.0)

    def test_gradients_recreates_on_each_access(self) -> None:
        """Accessing gradients twice returns different tensor objects."""
        x = torch.randn(1, 3)
        loss = self.linear(x).sum()
        loss.backward()

        flat1 = self.model.gradients
        flat2 = self.model.gradients

        self.assertIsNot(flat1, flat2)

    def test_repr(self) -> None:
        """__repr__ includes the module class name and parameter count."""
        r = repr(self.model)
        self.assertIn("Linear", r)
        self.assertIn("d=", r)

    def test_module_setter(self) -> None:
        """Setting a new module updates the internal reference."""
        new_module = nn.Linear(10, 5)
        self.model.module = new_module
        self.assertIs(self.model.module, new_module)

    def test_numel(self) -> None:
        """Numel returns the total number of scalar parameters."""
        expected = sum(p.numel() for p in self.linear.parameters())
        self.assertEqual(self.model.numel, expected)

    def test_set_gradients_writes_flat_into_grads(self) -> None:
        """set_gradients unpacks a flat tensor into each parameter's .grad."""
        d = self.model.numel
        flat = torch.randn(d)
        self.model.set_gradients(flat)
        for p in self.linear.parameters():
            self.assertIsNotNone(p.grad)
            self.assertEqual(p.grad.numel(), p.numel())
            self.assertFalse(torch.equal(p.grad, torch.zeros_like(p.grad)))


if __name__ == "__main__":
    unittest.main()
