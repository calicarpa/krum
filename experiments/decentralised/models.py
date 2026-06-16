"""Models for the MoNNA decentralised experiment."""

import torch
from torch import nn


class SmallMnistNet(nn.Module):
    """Small CPU-friendly classifier for 28x28 images."""

    def __init__(self) -> None:
        """Initialize the network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute logits."""
        return self.net(inputs)
