r"""MLP models for Byzantine-resilient distributed learning simulations.

These models implement the architectures used in the foundational papers on
Byzantine-resilient distributed learning. They are provided as ready-to-use
building blocks for simulations and can also serve as baselines for custom
experiments.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    r"""Two-layer MLP for MNIST classification.

    This architecture is used in both the NIPS 2017 and ICML 2018 papers
    for experiments on the MNIST dataset. The model has approximately
    :math:`8 \times 10^4` parameters.

    Architecture:

    .. math::

        \mathbb{R}^{784} \xrightarrow{\text{Linear}} \mathbb{R}^{100}
        \xrightarrow{\text{ReLU}} \mathbb{R}^{10}

    The model accepts both flattened inputs of shape ``(batch_size, 784)``
    and image tensors of shape ``(batch_size, 1, 28, 28)``.

    Example::

        from krum.primitives.models import MLP

        model = MLP()
        x = torch.randn(32, 784)  # batch of 32 flattened MNIST images
        output = model(x)          # shape: (32, 10)
    """

    def __init__(self) -> None:
        """Initialize the MLP model."""
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP model.

        Args:
            x: Input tensor of shape ``(batch_size, 784)`` or ``(batch_size, 1, 28, 28)``.

        Returns:
            Output tensor of shape ``(batch_size, 10)`` containing class logits.
        """
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MLPSpambase(nn.Module):
    r"""Two-hidden-layer MLP for Spambase classification.

    This architecture is used in the NIPS 2017 paper for experiments on the
    Spambase dataset (57 features, binary classification). The paper specifies
    an MLP with two hidden layers but does not give exact sizes. This
    implementation uses 20 and 20, consistent with reference implementations
    from the LPD-EPFL lab.

    Architecture:

    .. math::

        \mathbb{R}^{57} \xrightarrow{\text{Linear}} \mathbb{R}^{20}
        \xrightarrow{\text{ReLU}} \mathbb{R}^{20}
        \xrightarrow{\text{ReLU}} \mathbb{R}^{2}

    Example::

        from krum.primitives.models import MLPSpambase

        model = MLPSpambase()
        x = torch.randn(16, 57)  # batch of 16 Spambase samples
        output = model(x)         # shape: (16, 2)
    """

    def __init__(self) -> None:
        """Initialize the Spambase MLP model."""
        super().__init__()
        self.fc1 = nn.Linear(57, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Spambase MLP model.

        Args:
            x: Input tensor of shape ``(batch_size, 57)``.

        Returns:
            Output tensor of shape ``(batch_size, 2)`` containing class logits.
        """
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
