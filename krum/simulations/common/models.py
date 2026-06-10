"""Standard models shared across simulation protocols."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP for MNIST: 784 inputs → 100 (ReLU) → 10 outputs.

    Total ≈ 8 × 10⁴ parameters. Used by both NIPS 2017 and ICML 2018
    simulations (the architectures are identical in the papers).
    """

    def __init__(self) -> None:
        """Initialize the MLP model."""
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP model."""
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
