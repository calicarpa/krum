r"""CNN model for Byzantine-resilient distributed learning simulations.

This model implements the architecture used in the ICML 2018 paper for
experiments on the CIFAR-10 dataset. It is provided as a ready-to-use
building block for simulations and can also serve as a baseline for custom
experiments on image classification tasks.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    r"""Convolutional neural network for CIFAR-10 classification.

    This architecture is used in the ICML 2018 paper for experiments on the
    CIFAR-10 dataset. It consists of two convolutional layers followed by
    three fully connected layers, with ReLU activations and max pooling.

    Architecture:

    .. math::

        \mathbb{R}^{3 \times 32 \times 3}
        \xrightarrow{\text{Conv2d}(16)} \mathbb{R}^{16 \times 32 \times 32}
        \xrightarrow{\text{MaxPool}} \mathbb{R}^{16 \times 15 \times 15}
        \xrightarrow{\text{Conv2d}(64)} \mathbb{R}^{64 \times 15 \times 15}
        \xrightarrow{\text{MaxPool}} \mathbb{R}^{64 \times 6 \times 6}
        \xrightarrow{\text{FC}} \mathbb{R}^{384}
        \xrightarrow{\text{FC}} \mathbb{R}^{192}
        \xrightarrow{\text{FC}} \mathbb{R}^{10}

    All hidden layers use ReLU activation. The model expects input tensors
    of shape ``(batch_size, 3, 32, 32)``.

    Example::

        from krum.primitives.models import CNN

        model = CNN()
        x = torch.randn(16, 3, 32, 32)  # batch of 16 CIFAR-10 images
        output = model(x)                # shape: (16, 10)
    """

    def __init__(self) -> None:
        """Initialize the CNN model."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=4, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN model.

        Args:
            x: Input tensor of shape ``(batch_size, 3, 32, 32)``.

        Returns:
            Output tensor of shape ``(batch_size, 10)`` containing class logits.
        """
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=4, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
