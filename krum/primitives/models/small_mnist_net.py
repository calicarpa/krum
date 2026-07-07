"""Small MNIST model for decentralised learning simulations."""

import torch
import torch.nn as nn


class SmallMnistNet(nn.Module):
    r"""Small CPU-friendly classifier for 28x28 images.

    This is a lightweight MLP designed for MNIST classification in
    decentralised learning experiments. It uses a simple architecture
    with one hidden layer, making it suitable for quick experiments
    and CPU-based training.

    Architecture:

    .. math::

        \mathbb{R}^{28 \times 28} \xrightarrow{\text{Flatten}} \mathbb{R}^{784}
        \xrightarrow{\text{Linear}} \mathbb{R}^{128}
        \xrightarrow{\text{ReLU}} \mathbb{R}^{128}
        \xrightarrow{\text{Linear}} \mathbb{R}^{10}

    The model accepts both flattened inputs of shape ``(batch_size, 784)``
    and image tensors of shape ``(batch_size, 1, 28, 28)`` or ``(batch_size, 28, 28)``.

    Example::

        from krum.primitives.models import SmallMnistNet

        model = SmallMnistNet()
        x = torch.randn(32, 1, 28, 28)  # batch of 32 MNIST images
        output = model(x)                # shape: (32, 10)
    """

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
        """Compute logits.

        Args:
            inputs: Input tensor of shape ``(batch_size, 1, 28, 28)``,
                ``(batch_size, 28, 28)``, or ``(batch_size, 784)``.

        Returns:
            Output tensor of shape ``(batch_size, 10)`` containing class logits.
        """
        return self.net(inputs)
