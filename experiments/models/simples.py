# coding: utf-8
###
# @file   simples.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Collection of simple models.
###

"""Simple neural-network models used for small-scale experiments.

This module exposes four lightweight constructors — :func:`full`, :func:`conv`,
:func:`logit` and :func:`linear` — that can be registered automatically by the
:class:`experiments.Model` loader because they are listed in ``__all__``.
Each constructor returns a ready-to-use ``torch.nn.Module``.

Example
-------
>>> from experiments import Model, Configuration
>>> config = Configuration(device="cpu")
>>> model = Model("simples-full", config)
>>> output = model.run(torch.randn(4, 1, 28, 28))
"""

__all__ = ["full", "conv", "logit", "linear"]

import torch


# ---------------------------------------------------------------------------- #
# Simple fully-connected model, for MNIST


class _Full(torch.nn.Module):
    """Small fully-connected classifier for MNIST.

    The network flattens a 28×28 input image, passes it through a single
    hidden layer of 100 units with ReLU activation, and finally outputs
    log-probabilities over 10 classes.
    """

    def __init__(self):
        """Initialise the two linear layers."""
        super().__init__()
        self._f1 = torch.nn.Linear(28 * 28, 100)
        self._f2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, 1, 28, 28)`` or ``(N, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Log-probability distribution of shape ``(N, 10)``.
        """
        x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
        x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
        return x


def full(*args, **kwargs):
    """Build a small fully-connected model for MNIST.

    Parameters
    ----------
    *args : object
        Forwarded to :class:`_Full`.
    **kwargs : object
        Forwarded to :class:`_Full`.

    Returns
    -------
    _Full
        A fresh fully-connected model instance.
    """
    return _Full(*args, **kwargs)


# ---------------------------------------------------------------------------- #
# Simple convolutional model, for MNIST


class _Conv(torch.nn.Module):
    """Small convolutional classifier for MNIST.

    The architecture consists of two convolutional blocks
    (convolution → ReLU → max-pooling) followed by two fully-connected
    layers. It expects single-channel 28×28 images and produces
    log-probabilities over 10 classes.
    """

    def __init__(self):
        """Initialise the convolutional and linear layers."""
        super().__init__()
        self._c1 = torch.nn.Conv2d(1, 20, 5, 1)
        self._c2 = torch.nn.Conv2d(20, 50, 5, 1)
        self._f1 = torch.nn.Linear(800, 500)
        self._f2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Log-probability distribution of shape ``(N, 10)``.
        """
        x = torch.nn.functional.relu(self._c1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self._c2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self._f1(x.view(-1, 800)))
        x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
        return x


def conv(*args, **kwargs):
    """Build a small convolutional model for MNIST.

    Parameters
    ----------
    *args : object
        Forwarded to :class:`_Conv`.
    **kwargs : object
        Forwarded to :class:`_Conv`.

    Returns
    -------
    _Conv
        A fresh convolutional model instance.
    """
    return _Conv(*args, **kwargs)


# ---------------------------------------------------------------------------- #
# Logistic regression model


class _Logit(torch.nn.Module):
    """Logistic regression model.

    A single linear layer followed by a sigmoid. Useful for binary
    classification or as a simple baseline.
    """

    def __init__(self, din, dout=1):
        """Initialise the linear layer.

        Parameters
        ----------
        din : int
            Number of input features.
        dout : int, optional
            Number of output features. Defaults to ``1``.
        """
        super().__init__()
        self._din = din
        self._dout = dout
        self._linear = torch.nn.Linear(din, dout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape; the last dimensions are
            flattened to ``din`` features.

        Returns
        -------
        torch.Tensor
            Sigmoid-activated output of shape ``(..., dout)``.
        """
        return torch.sigmoid(self._linear(x.view(-1, self._din)))


def logit(*args, **kwargs):
    """Build a logistic-regression model.

    Parameters
    ----------
    *args : object
        Forwarded to :class:`_Logit`.
    **kwargs : object
        Forwarded to :class:`_Logit`.

    Returns
    -------
    _Logit
        A fresh logistic-regression model instance.
    """
    return _Logit(*args, **kwargs)


# ---------------------------------------------------------------------------- #
# Linear regression model


class _Linear(torch.nn.Module):
    """Simple linear (affine) model without activation.

    Equivalent to a fully-connected layer with identity activation.
    """

    def __init__(self, din, dout=1):
        """Initialise the linear layer.

        Parameters
        ----------
        din : int
            Number of input features.
        dout : int, optional
            Number of output features. Defaults to ``1``.
        """
        super().__init__()
        self._din = din
        self._dout = dout
        self._linear = torch.nn.Linear(din, dout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape; the last dimensions are
            flattened to ``din`` features.

        Returns
        -------
        torch.Tensor
            Linear output of shape ``(..., dout)``.
        """
        return self._linear(x.view(-1, self._din))


def linear(*args, **kwargs):
    """Build a simple linear model.

    Parameters
    ----------
    *args : object
        Forwarded to :class:`_Linear`.
    **kwargs : object
        Forwarded to :class:`_Linear`.

    Returns
    -------
    _Linear
        A fresh linear model instance.
    """
    return _Linear(*args, **kwargs)
