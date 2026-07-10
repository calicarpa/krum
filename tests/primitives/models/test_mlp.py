"""Tests for MLP models."""

import unittest

import torch

from krum.primitives.models import (
    Krum2017MLPMnist,
    Krum2017MLPSpambase,
    Monna2023SmallMnist,
)


class Krum2017MLPMnistTest(unittest.TestCase):
    """Test the Krum2017MLPMnist model for MNIST."""

    def setUp(self):
        """Set up a Krum2017MLPMnist model for testing."""
        self.model = Krum2017MLPMnist()

    def test_forward_pass_flat_input(self):
        """Forward pass works with flat input of shape (batch_size, 784)."""
        x = torch.randn(4, 784)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_image_input(self):
        """Forward pass works with image input of shape (batch_size, 1, 28, 28)."""
        x = torch.randn(4, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_single_sample(self):
        """Forward pass works with a single sample."""
        x = torch.randn(1, 784)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_output_dtype_matches_input(self):
        """Output dtype matches input dtype."""
        x = torch.randn(2, 784, dtype=torch.float32)
        output = self.model(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_gradient_flow(self):
        """Gradients flow through the model."""
        x = torch.randn(2, 784)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))  # ty:ignore[no-matching-overload]


class Krum2017MLPSpambaseTest(unittest.TestCase):
    """Test the Krum2017MLPSpambase model for Spambase dataset."""

    def setUp(self):
        """Set up a Krum2017MLPSpambase model for testing."""
        self.model = Krum2017MLPSpambase()

    def test_forward_pass(self):
        """Forward pass works with input of shape (batch_size, 57)."""
        x = torch.randn(8, 57)
        output = self.model(x)
        self.assertEqual(output.shape, (8, 2))

    def test_forward_pass_single_sample(self):
        """Forward pass works with a single sample."""
        x = torch.randn(1, 57)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 2))

    def test_output_dtype_matches_input(self):
        """Output dtype matches input dtype."""
        x = torch.randn(4, 57, dtype=torch.float32)
        output = self.model(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_gradient_flow(self):
        """Gradients flow through the model."""
        x = torch.randn(4, 57)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))  # ty:ignore[no-matching-overload]


class Monna2023SmallMnistNetTest(unittest.TestCase):
    """Test the Monna2023SmallMnist model for MNIST."""

    def setUp(self):
        """Set up a Monna2023SmallMnist model for testing."""
        self.model = Monna2023SmallMnist()

    def test_forward_pass_flat_input(self):
        """Forward pass works with flat input of shape (batch_size, 784)."""
        x = torch.randn(4, 784)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_image_input(self):
        """Forward pass works with image input of shape (batch_size, 1, 28, 28)."""
        x = torch.randn(4, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_2d_image_input(self):
        """Forward pass works with 2D image input of shape (batch_size, 28, 28)."""
        x = torch.randn(4, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_single_sample(self):
        """Forward pass works with a single sample."""
        x = torch.randn(1, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_output_dtype_matches_input(self):
        """Output dtype matches input dtype."""
        x = torch.randn(2, 1, 28, 28, dtype=torch.float32)
        output = self.model(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_gradient_flow(self):
        """Gradients flow through the model."""
        x = torch.randn(2, 1, 28, 28)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))  # ty:ignore[no-matching-overload]


if __name__ == "__main__":
    unittest.main()
