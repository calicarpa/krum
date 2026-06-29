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
        self.assertEqual(self.linear.weight[0, 0].item(), 99.0)

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

    def test_parameters_setter_relinks_weights(self) -> None:
        """Setting parameters unpacks a flat tensor into the module weights."""
        d = sum(p.numel() for p in self.linear.parameters())
        flat = torch.randn(d)
        self.model.parameters = flat
        self.assertIs(self.model.parameters, flat)

    def test_gradients_zero_copy(self) -> None:
        """Modifying the flat gradients tensor updates individual grads."""
        x = torch.randn(1, 3)
        loss = self.linear(x).sum()
        loss.backward()

        flat_grads = self.model.gradients
        flat_grads[0] = 99.0

        grad = self.linear.weight.grad
        assert grad is not None
        self.assertEqual(grad[0, 0].item(), 99.0)

    def test_gradients_cache_returns_same_object(self) -> None:
        """Accessing gradients twice returns the same tensor object."""
        x = torch.randn(1, 3)
        loss = self.linear(x).sum()
        loss.backward()

        flat1 = self.model.gradients
        flat2 = self.model.gradients

        self.assertIs(flat1, flat2)

    def test_gradients_setter_writes_flat_into_grads(self) -> None:
        """Setting gradients unpacks a flat tensor into each parameter's .grad."""
        d = sum(p.numel() for p in self.linear.parameters())
        flat = torch.randn(d)
        self.model.gradients = flat
        for p in self.linear.parameters():
            grad = p.grad
            assert grad is not None
            self.assertEqual(grad.numel(), p.numel())
            self.assertFalse(torch.equal(grad, torch.zeros_like(grad)))

    def test_module_setter(self) -> None:
        """Setting a new module updates the internal reference."""
        new_module = nn.Linear(10, 5)
        self.model.module = new_module
        self.assertIs(self.model.module, new_module)

    def test_repr(self) -> None:
        """__repr__ includes the module class name."""
        r = repr(self.model)
        self.assertIn("Linear", r)

    def test_slots_prevent_arbitrary_attributes(self) -> None:
        """Cannot set arbitrary attributes on slotted instances."""
        with self.assertRaises(AttributeError):
            self.model.extra = 42  # ty:ignore[unresolved-attribute]

    def _forward_backward(self) -> None:
        x = torch.randn(1, 3)
        loss = self.linear(x).sum()
        loss.backward()

    def test_relink_gradients_before_first_access(self) -> None:
        """relink_gradients works without prior access to .gradients (lazy init)."""
        result = self.model.relink_gradients()
        self.assertIsNotNone(result)
        self.assertIs(result, self.model.gradients)

    def test_relink_gradients_after_zero_grad_set_to_none(self) -> None:
        """After zero_grad(set_to_none=True), relink_gradients restores zero-copy."""
        self._forward_backward()
        flat = self.model.gradients

        self.linear.zero_grad(set_to_none=True)
        self.assertIsNone(self.linear.weight.grad)
        self.assertIsNone(self.linear.bias.grad)

        self.model.relink_gradients()
        self.assertIsNotNone(self.linear.weight.grad)
        self.assertIsNotNone(self.linear.bias.grad)

        flat[0] = 77.0
        wg = self.linear.weight.grad
        assert wg is not None
        self.assertEqual(wg[0, 0].item(), 77.0)

    def test_relink_gradients_returns_flat_tensor(self) -> None:
        """relink_gradients returns the flat gradient tensor, enabling single-call access."""
        self._forward_backward()
        result = self.model.relink_gradients()
        self.assertIs(result, self.model.gradients)

    def test_relink_parameters_returns_flat_tensor(self) -> None:
        """relink_parameters returns the flat parameter tensor, enabling single-call access."""
        result = self.model.relink_parameters()
        self.assertIs(result, self.model.parameters)

    def test_relink_gradients_preserves_grad_values(self) -> None:
        """relink_gradients copies existing gradient values into the flat buffer."""
        self._forward_backward()
        _flat = self.model.gradients

        wg = self.linear.weight.grad
        assert wg is not None
        replacement = wg.clone().add_(1.0)
        self.linear.weight.grad = replacement
        self.model.relink_gradients()

        torch.testing.assert_close(self.linear.weight.grad, replacement)

    def test_relink_gradients_preserves_tensor_instance(self) -> None:
        """relink_gradients keeps the user's .grad Tensor instance after relinking."""
        self._forward_backward()
        _flat = self.model.gradients

        new_grad = torch.randn_like(self.linear.weight)
        self.linear.weight.grad = new_grad

        self.model.relink_gradients()
        self.assertIs(self.linear.weight.grad, new_grad)

        flat = self.model.gradients
        flat[0] = 99.0
        self.assertEqual(new_grad[0, 0].item(), 99.0)

    def test_relink_gradients_noop_when_grads_already_linked(self) -> None:
        """relink_gradients is a no-op when all grads already share the flat buffer."""
        self._forward_backward()
        flat = self.model.gradients

        self.model.relink_gradients()

        flat[0] = 88.0
        wg = self.linear.weight.grad
        assert wg is not None
        self.assertEqual(wg[0, 0].item(), 88.0)

    def test_relink_parameters_before_first_access(self) -> None:
        """relink_parameters works without prior access to .parameters (lazy init)."""
        result = self.model.relink_parameters()
        self.assertIsNotNone(result)
        self.assertIs(result, self.model.parameters)

    def test_relink_parameters_after_data_replacement(self) -> None:
        """After replacing a parameter's .data, relink_parameters restores zero-copy."""
        _flat = self.model.parameters

        with torch.no_grad():
            self.linear.weight.data = torch.randn_like(self.linear.weight)

        self.model.relink_parameters()

        flat = self.model.parameters
        flat[0] = 55.0
        self.assertEqual(self.linear.weight[0, 0].item(), 55.0)

    def test_relink_parameters_preserves_param_values(self) -> None:
        """relink_parameters copies existing parameter values into the flat buffer."""
        _flat = self.model.parameters
        w = self.linear.weight.data
        replacement = w.clone().add_(1.0)

        with torch.no_grad():
            self.linear.weight.data = replacement

        self.model.relink_parameters()
        torch.testing.assert_close(self.linear.weight.data, replacement)

    def test_relink_parameters_preserves_tensor_instance(self) -> None:
        """relink_parameters preserves values but replaces the .data tensor instance."""
        _flat = self.model.parameters
        new_data = torch.randn_like(self.linear.weight)

        with torch.no_grad():
            self.linear.weight.data = new_data

        self.model.relink_parameters()
        self.assertIsNot(self.linear.weight.data, new_data)
        torch.testing.assert_close(self.linear.weight.data, new_data)

        flat = self.model.parameters
        flat[0] = 99.0
        self.assertEqual(self.linear.weight[0, 0].item(), 99.0)

    def test_relink_parameters_noop_when_already_linked(self) -> None:
        """relink_parameters is a no-op when all params already share the flat buffer."""
        _flat = self.model.parameters
        self.model.relink_parameters()

        flat = self.model.parameters
        flat[0] = 88.0
        self.assertEqual(self.linear.weight[0, 0].item(), 88.0)

    def test_relink_restores_after_combined_replacement(self) -> None:
        """relink_gradients and relink_parameters work independently after external replacement."""
        self._forward_backward()
        _flat_p = self.model.parameters
        _flat_g = self.model.gradients

        self.linear.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.linear.weight.data = torch.randn_like(self.linear.weight)

        flat_g = self.model.relink_gradients()
        flat_p = self.model.relink_parameters()

        flat_g[0] = 77.0
        wg = self.linear.weight.grad
        assert wg is not None
        self.assertEqual(wg[0, 0].item(), 77.0)

        flat_p[0] = 33.0
        self.assertEqual(self.linear.weight[0, 0].item(), 33.0)


if __name__ == "__main__":
    unittest.main()
