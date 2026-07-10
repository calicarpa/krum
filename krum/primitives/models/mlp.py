r"""MLP models for Byzantine-resilient distributed learning simulations.

These models implement the architectures used in the foundational papers on
Byzantine-resilient distributed learning and decentralised learning. They are
provided as ready-to-use building blocks for simulations and can also serve as
baselines for custom experiments.
"""

import torch.nn as nn


class Krum2017MLPMnist(nn.Sequential):
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

        from krum.primitives.models import Krum2017MLPMnist

        model = Krum2017MLPMnist()
        x = torch.randn(32, 784)  # batch of 32 flattened MNIST images
        output = model(x)          # shape: (32, 10)
    """

    def __init__(self) -> None:
        """Initialize the MNIST MLP model."""
        super().__init__(
            nn.Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )


class Krum2017MLPSpambase(nn.Sequential):
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

        from krum.primitives.models import Krum2017MLPSpambase

        model = Krum2017MLPSpambase()
        x = torch.randn(16, 57)  # batch of 16 Spambase samples
        output = model(x)         # shape: (16, 2)
    """

    def __init__(self) -> None:
        """Initialize the Spambase MLP model."""
        super().__init__(
            nn.Flatten(),
            nn.Linear(57, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )


class Monna2023SmallMnistNet(nn.Sequential):
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

        from krum.primitives.models import Monna2023SmallMnistNet

        model = Monna2023SmallMnistNet()
        x = torch.randn(32, 1, 28, 28)  # batch of 32 MNIST images
        output = model(x)                # shape: (32, 10)
    """

    def __init__(self) -> None:
        """Initialize the network."""
        super().__init__(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
