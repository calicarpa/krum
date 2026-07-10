"""Tests for Krum2017CNN model."""

import unittest

import torch

from krum.primitives.models import Krum2017CNN


class Krum2017CNNTest(unittest.TestCase):
    """Test the Krum2017CNN model for CIFAR-10."""

    def setUp(self):
        """Set up a Krum2017CNN model for testing."""
        self.model = Krum2017CNN()

    def test_forward_pass(self):
        """Forward pass works with input of shape (batch_size, 3, 32, 32)."""
        x = torch.randn(4, 3, 32, 32)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_single_sample(self):
        """Forward pass works with a single sample."""
        x = torch.randn(1, 3, 32, 32)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_output_dtype_matches_input(self):
        """Output dtype matches input dtype."""
        x = torch.randn(2, 3, 32, 32, dtype=torch.float32)
        output = self.model(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_gradient_flow(self):
        """Gradients flow through the model."""
        x = torch.randn(2, 3, 32, 32)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))  # ty:ignore[no-matching-overload]


if __name__ == "__main__":
    unittest.main()
