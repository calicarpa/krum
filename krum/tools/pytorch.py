###
# @file   pytorch.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Helpers relative to PyTorch.
###

"""
This module provides helper functions for PyTorch tensor manipulation,
gradient handling, and common operations used throughout Krum.

Functions
---------

**Memory Management:**

- ``relink``: Make tensors point to a contiguous memory segment
- ``flatten``: Flatten tensors into a single contiguous tensor

**Gradient Operations:**

- ``grad_of``: Get or create gradient for a tensor
- ``grads_of``: Generator version for multiple tensors

**Statistics:**

- ``compute_avg_dev_max``: Compute mean, std, norm stats

**Time Measurement:**

- ``AccumulatedTimedContext``: Accumulated timing with optional CUDA sync

**Utilities:**

- ``weighted_mse_loss``: Weighted MSE loss for experiments
- ``regression``: Generic optimization for free variables
- ``pnm``: Export tensor to PGM/PBM format

Example
-------

.. code-block:: python

    import torch
    from tools import flatten, relink

    # Flatten model parameters
    params = list(model.parameters())
    flat_params = flatten(params)

    # Relink gradients to same memory
    grads = [p.grad for p in params]
    flat_grads = flatten(grads)
"""

__all__ = [
    "AccumulatedTimedContext",
    "WeightedMSELoss",
    "compute_avg_dev_max",
    "flatten",
    "grad_of",
    "grads_of",
    "pnm",
    "regression",
    "relink",
    "weighted_mse_loss",
]

import io
import time
import types
from collections.abc import Callable

import torch

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations


def relink(tensors: list[torch.Tensor], common: torch.Tensor) -> torch.Tensor:
    """
    Relink tensors to share a common contiguous memory storage.

    Parameters
    ----------
    tensors : iterable of torch.Tensor
        Tensors to relink. All must have the same dtype.
    common : torch.Tensor
        Flat tensor of sufficient size to use as underlying storage.
        Must have the same dtype as the given tensors.

    Returns
    -------
    torch.Tensor
        The common tensor, with ``linked_tensors`` attribute set.

    Notes
    -----
    The returned tensor has a ``linked_tensors`` attribute pointing to the
    original tensors. This allows updating all tensors simultaneously.

    Example
    -------

    >>> import torch
    >>> from tools import relink
    >>> t1 = torch.tensor([1., 2.])
    >>> t2 = torch.tensor([3., 4., 5.])
    >>> common = torch.zeros(5)
    >>> relink([t1, t2], common)
    tensor([1., 2., 3., 4., 5.])
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Relink each given tensor to its segment on the common one
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        tensor.data = common[pos:npos].view(*tensor.shape)
        pos = npos
    # Finalize and return
    common.linked_tensors = tensors
    return common


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Flatten tensors into a single contiguous tensor.

    Parameters
    ----------
    tensors : iterable of torch.Tensor
        Tensors to flatten. All must have the same dtype.

    Returns
    -------
    torch.Tensor
        Flat tensor containing all data from input tensors, stored in
        a contiguous memory segment.

    Notes
    -----
    The returned tensor shares memory with the original tensors. Modifications
    to the flat tensor will reflect in the original tensors.

    Example
    -------

    >>> import torch
    >>> from tools import flatten
    >>> t1 = torch.tensor([1., 2.])
    >>> t2 = torch.tensor([3., 4., 5.])
    >>> flat = flatten([t1, t2])
    tensor([1., 2., 3., 4., 5.])
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Common tensor instantiation and reuse
    common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
    # Return common tensor
    return relink(tensors, common)


# ---------------------------------------------------------------------------- #
# Gradient access


def grad_of(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the gradient of a given tensor, create zero gradient if missing.

    Parameters
    ----------
    tensor : torch.Tensor
        A tensor that may have a gradient attached.

    Returns
    -------
    torch.Tensor
        The gradient tensor. If none existed, a zero gradient is created
        and attached to the tensor.

    Example
    -------

    >>> import torch
    >>> from tools import grad_of
    >>> x = torch.randn(3, requires_grad=True)
    >>> y = x.sum()
    >>> y.backward()
    >>> grad = grad_of(x)
    """
    # Get the current gradient
    grad = tensor.grad
    if grad is not None:
        return grad
    # Make and set a zero-gradient
    grad = torch.zeros_like(tensor)
    tensor.grad = grad
    return grad


def grads_of(tensors: list[torch.Tensor]):
    """
    Generator that gets or creates gradients for multiple tensors.

    Parameters
    ----------
    tensors : iterable of torch.Tensor
        Tensors that may have gradients attached.

    Yields
    ------
    torch.Tensor
        Gradient for each tensor.

    Example
    -------

    >>> import torch
    >>> from tools import grads_of
    >>> params = [torch.randn(3, requires_grad=True) for _ in range(2)]
    >>> loss = sum(p.sum() for p in params)
    >>> loss.backward()
    >>> for g in grads_of(params):
    ...     print(g)
    tensor([1., 1., 1.])
    tensor([1., 1., 1.])
    """
    for tensor in tensors:
        yield grad_of(tensor)


# ---------------------------------------------------------------------------- #
# Statistics


def compute_avg_dev_max(
    samples: list[torch.Tensor],
) -> tuple[torch.Tensor | None, float, float, float]:
    """
    Compute average, average norm, norm deviation, and max absolute value.

    Parameters
    ----------
    samples : list of torch.Tensor
        List of tensors to compute statistics on.

    Returns
    -------
    tuple[torch.Tensor, float, float, float]
        Tuple containing: average tensor, average norm, norm deviation, and
        max absolute value.

    Notes
    -----
    The returned tensor is newly created and does not alias any input tensor.
    """
    # Handle empty list gracefully
    if len(samples) == 0:
        return None, float("nan"), float("nan"), float("nan")
    # Stack all samples
    stacked = torch.stack(samples)
    # Compute average tensor
    avg = stacked.mean(dim=0)
    # Compute norms
    norms = stacked.norm(dim=1)
    # Average norm and deviation
    avg_norm = norms.mean().item()
    dev_norm = norms.std().item() if len(norms) > 1 else 0.0
    # Max absolute value across all samples
    max_abs = stacked.abs().max().item()
    return avg, avg_norm, dev_norm, max_abs


# ---------------------------------------------------------------------------- #
# Accumulated timed context


class AccumulatedTimedContext:
    """
    Accumulated timed context manager with optional CUDA synchronization.

    This context manager measures elapsed time across multiple entries,
    with optional CUDA synchronization to ensure accurate GPU timing.

    Parameters
    ----------
    sync : bool, optional
        Whether to synchronize CUDA before and after timing. Defaults to
        ``False``.

    Example
    -------
    >>> import torch
    >>> from tools import AccumulatedTimedContext
    >>> atc = AccumulatedTimedContext(sync=True)
    >>> with atc:
    ...     # GPU operations here
    ...     pass
    >>> print(atc.current_runtime())
    """

    def __init__(self, sync: bool = False) -> None:
        """
        Initialize the accumulated timed context.

        Parameters
        ----------
        sync : bool, optional
            Whether to synchronize CUDA before and after timing. Defaults to
            ``False``.
        """
        self._sync = sync
        self._start = None
        self._elapsed = 0.0

    def __enter__(self):
        """
        Enter the context and start timing.

        Returns
        -------
        AccumulatedTimedContext
            Self reference for context management.
        """
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the context, stop timing, and accumulate elapsed time.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to the context exit.
        """
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._elapsed += time.perf_counter() - self._start

    def current_runtime(self) -> float:
        """
        Return the accumulated runtime.

        Returns
        -------
        float
            Total accumulated time in seconds.
        """
        return self._elapsed


# ---------------------------------------------------------------------------- #
# Weighted MSE loss


def weighted_mse_loss(
    input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Compute weighted mean squared error loss.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor.
    weight : torch.Tensor
        Weight tensor for each element.

    Returns
    -------
    torch.Tensor
        Weighted MSE loss value.

    Notes
    -----
    The returned tensor is newly created and does not alias any input tensor.
    """
    return (weight * (input - target) ** 2).mean()


class WeightedMSELoss(torch.nn.Module):
    """
    Weighted MSE loss module.

    This module wraps :func:`weighted_mse_loss` as a PyTorch module.
    """

    def __init__(self) -> None:
        """
        Initialize the weighted MSE loss module.
        """
        super().__init__()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.
        target : torch.Tensor
            Target tensor.
        weight : torch.Tensor
            Weight tensor.

        Returns
        -------
        torch.Tensor
            Weighted MSE loss value.
        """
        return weighted_mse_loss(input, target, weight)


# ---------------------------------------------------------------------------- #
# Regression helper


def regression(
    func: Callable[[torch.Tensor, dict], torch.Tensor],
    vars,
    data,
    loss=None,
    opt=None,
    steps=1000,
) -> float:
    """
    Generic optimization for free variables.

    Parameters
    ----------
    func : callable
        Function to optimize. Takes variables and data dictionary as arguments.
    vars : list
        List of variables to optimize.
    data : dict
        Data dictionary, must contain a ``"target"`` key.
    loss : torch.nn.Module, optional
        Loss function. Defaults to ``torch.nn.MSELoss``.
    opt : torch.optim.Optimizer, optional
        Optimizer. Defaults to ``torch.optim.Adam``.
    steps : int, optional
        Number of optimization steps. Defaults to 1000.

    Returns
    -------
    float
        Final loss value after optimization.
    """
    if loss is None:
        loss = torch.nn.MSELoss()
    if opt is None:
        opt = torch.optim.Adam(vars)
    for _ in range(steps):
        opt.zero_grad()
        result = func(vars, data)
        loss_func = loss(result, data["target"])
        loss_func.backward()
        opt.step()
    return loss_func.item()


# ---------------------------------------------------------------------------- #
# PNM export


def pnm(fd: io.BufferedWriter, tn: torch.Tensor) -> None:
    """
    Export tensor to PGM/PBM format.

    Parameters
    ----------
    fd : io.BufferedWriter
        File descriptor to write to.
    tn : torch.Tensor
        Tensor to export. Supports float32/float64 for grayscale (PGM) or
        boolean/integer for binary (PBM).

    Notes
    -----
    - Grayscale format (PGM): For float32/float64 tensors, normalizes to 0-255.
    - Binary format (PBM): For other dtypes, converts to binary values.
    """
    if tn.dtype == torch.float32 or tn.dtype == torch.float64:
        # Grayscale
        m = tn.min().item()
        M = tn.max().item()
        if M - m < 1e-8:
            M = m + 1
        t = ((tn - m) / (M - m) * 255).byte().cpu()
        fd.write(f"P5\n{tn.shape[1]}\n{tn.shape[0]}\n255\n")
        fd.write(t.numpy().tobytes())
    else:
        # Binary
        t = (tn > 0).byte().cpu()
        fd.write(f"P4\n{tn.shape[1]}\n{tn.shape[0]}\n")
        # Pad to byte boundary
        w = (tn.shape[1] + 7) // 8
        pad = w * 8 - tn.shape[1]
        for row in t:
            row = torch.cat([row, torch.zeros(pad, dtype=torch.uint8)])
            fd.write(row.view(-1).numpy().tobytes())
