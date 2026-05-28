"""Models for the Krum-NIPS-2017 simulation."""

import torch
import torch.nn as nn


class MLPMnist(nn.Module):
    """MLP for MNIST in the Krum-NIPS-2017 simulation.

    Architecture: 784 inputs, 1 hidden layer of size 100 (ReLU),
    10 outputs. Total ≈ 8 × 10⁴ parameters.
    """

    def __init__(self):
        """Initialize the MLP model."""
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP model.

        Args:
            x: Input tensor of shape (batch_size, 784).

        Returns:
            Output tensor of shape (batch_size, 10).
        """
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MLPSpambase(nn.Module):
    """MLP with two hidden layers for Spambase in the Krum-NIPS-2017 simulation.

    The paper specifies an MLP with two hidden layers but does not give
    exact sizes. This implementation uses 20 and 20, which is consistent
    with reference implementations from the LPD-EPFL lab.

    Architecture: 57 inputs → 20 (ReLU) → 20 (ReLU) → 2 outputs.
    """

    def __init__(self):
        """Initialize the Spambase MLP model."""
        super().__init__()
        self.fc1 = nn.Linear(57, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Spambase MLP model.

        Args:
            x: Input tensor of shape (batch_size, 57).

        Returns:
            Output tensor of shape (batch_size, 2).
        """
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
