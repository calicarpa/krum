r"""CNN models for Byzantine-resilient distributed learning simulations.

These models implement the architectures used in the foundational papers on
Byzantine-resilient distributed learning and decentralised learning. They are
provided as ready-to-use building blocks for simulations and can also serve as
baselines for custom experiments on image classification tasks.
"""

import torch.nn as nn


class Krum2017CNN(nn.Sequential):
    r"""Convolutional neural network for CIFAR-10 classification.

    This architecture is used in the ICML 2018 paper for experiments on the
    CIFAR-10 dataset. It consists of two convolutional layers followed by
    three fully connected layers, with ReLU activations and max pooling.

    Architecture:

    .. math::

        \mathbb{R}^{3 \times 32 \times 32}
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

        from krum.primitives.models import Krum2017CNN

        model = Krum2017CNN()
        x = torch.randn(16, 3, 32, 32)  # batch of 16 CIFAR-10 images
        output = model(x)                # shape: (16, 10)
    """

    def __init__(self) -> None:
        """Initialize the CNN model."""
        super().__init__(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10),
        )


class Monna2023CNNMnist(nn.Sequential):
    r"""Convolutional neural network for MNIST classification.

    This architecture is used in the ICML 2023 paper for experiments on the
    MNIST dataset (Table 2, Appendix D.2): ``C(20)-R-M-C(20)-R-M-L(500)-R-L(10)``
    — two convolutional layers followed by two fully connected layers, with
    ReLU activations and max pooling. The paper trains this model with
    learning rate :math:`\gamma = 0.75` for :math:`T = 600` iterations, over
    :math:`n = 26` nodes of which :math:`f = 5` are Byzantine.

    Unlike the paper, this model outputs raw logits rather than
    log-probabilities (no final log-softmax), for consistency with the other
    models in this module and to pair directly with ``nn.CrossEntropyLoss``
    (equivalent to the paper's log-softmax + NLL-loss pairing).

    Architecture:

    .. math::

        \mathbb{R}^{1 \times 28 \times 28}
        \xrightarrow{\text{Conv2d}(20)} \mathbb{R}^{20 \times 24 \times 24}
        \xrightarrow{\text{MaxPool}} \mathbb{R}^{20 \times 12 \times 12}
        \xrightarrow{\text{Conv2d}(20)} \mathbb{R}^{20 \times 8 \times 8}
        \xrightarrow{\text{MaxPool}} \mathbb{R}^{20 \times 4 \times 4}
        \xrightarrow{\text{FC}} \mathbb{R}^{500}
        \xrightarrow{\text{FC}} \mathbb{R}^{10}

    Both convolutions use kernel size 5, stride 1, and no padding. All hidden
    layers use ReLU activation. The model expects input tensors of shape
    ``(batch_size, 1, 28, 28)``.

    Example::

        from krum.primitives.models import Monna2023CNNMnist

        model = Monna2023CNNMnist()
        x = torch.randn(16, 1, 28, 28)  # batch of 16 MNIST images
        output = model(x)                # shape: (16, 10)
    """

    def __init__(self) -> None:
        """Initialize the CNN model."""
        super().__init__(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(20 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )


class Monna2023CNNCifar10(nn.Sequential):
    r"""Convolutional neural network for CIFAR-10 classification.

    This architecture is used in the ICML 2023 paper for experiments on the
    CIFAR-10 dataset (Table 2, Appendix D.2):
    ``C(64)-R-B-C(64)-R-B-M-D-C(128)-R-B-C(128)-R-B-M-D-L(128)-R-D-L(10)`` —
    four convolutional layers followed by two fully connected layers, with
    ReLU activations, batch normalization, max pooling, and dropout. The paper
    trains this model with learning rate :math:`\gamma = 0.5` for
    :math:`T = 2000` iterations, over :math:`n = 16` nodes of which
    :math:`f = 3` are Byzantine.

    Unlike the paper, this model outputs raw logits rather than
    log-probabilities (no final log-softmax), for consistency with the other
    models in this module and to pair directly with ``nn.CrossEntropyLoss``
    (equivalent to the paper's log-softmax + NLL-loss pairing).

    Architecture:

    .. math::

        \mathbb{R}^{3 \times 32 \times 32}
        \xrightarrow{\text{Conv2d}(64)} \mathbb{R}^{64 \times 28 \times 28}
        \xrightarrow{\text{Conv2d}(64)} \mathbb{R}^{64 \times 24 \times 24}
        \xrightarrow{\text{MaxPool}} \mathbb{R}^{64 \times 12 \times 12}
        \xrightarrow{\text{Conv2d}(128)} \mathbb{R}^{128 \times 8 \times 8}
        \xrightarrow{\text{Conv2d}(128)} \mathbb{R}^{128 \times 4 \times 4}
        \xrightarrow{\text{MaxPool}} \mathbb{R}^{128 \times 2 \times 2}
        \xrightarrow{\text{FC}} \mathbb{R}^{128}
        \xrightarrow{\text{FC}} \mathbb{R}^{10}

    Every convolution uses kernel size 5, stride 1, and no padding. Each
    convolutional layer is followed by ReLU and batch normalization; each
    max-pooling stage is followed by dropout (:math:`p = 0.25`), as is the
    first fully connected layer. The model expects input tensors of shape
    ``(batch_size, 3, 32, 32)``.

    Example::

        from krum.primitives.models import Monna2023CNNCifar10

        model = Monna2023CNNCifar10()
        x = torch.randn(16, 3, 32, 32)  # batch of 16 CIFAR-10 images
        output = model(x)                # shape: (16, 10)
    """

    def __init__(self) -> None:
        """Initialize the CNN model."""
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )
