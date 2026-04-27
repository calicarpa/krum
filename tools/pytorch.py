# coding: utf-8
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

__all__ = ["relink", "flatten", "grad_of", "grads_of", "compute_avg_dev_max",
           "AccumulatedTimedContext", "weighted_mse_loss", "WeightedMSELoss",
           "regression", "pnm"]

import math
import time
import torch
import types

import tools

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
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

def flatten(tensors):
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

def grad_of(tensor):
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

def grads_of(tensors):
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

def compute_avg_dev_max(samples):
    """ Compute average, average norm, norm deviation, and max absolute value.
    Args:
        samples: List of tensors
    Returns:
        Tuple of (average tensor, average norm, norm deviation, max abs)
    """
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
    """ Accumulated timed context manager with optional CUDA synchronization.

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

    def __init__(self, sync=False):
        self._sync = sync
        self._start = None
        self._elapsed = 0.0

    def __enter__(self):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._elapsed += time.perf_counter() - self._start

    def current_runtime(self):
        return self._elapsed

# ---------------------------------------------------------------------------- #
# Weighted MSE loss

def weighted_mse_loss(input, target, weight):
    """ Weighted MSE loss.
    Args:
        input: Input tensor
        target: Target tensor
        weight: Weight tensor
    Returns:
        Weighted MSE loss
    """
    return (weight * (input - target) ** 2).mean()

class WeightedMSELoss(torch.nn.Module):
    """ Weighted MSE loss module. """

    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight):
        return weighted_mse_loss(input, target, weight)

# ---------------------------------------------------------------------------- #
# Regression helper

def regression(func, vars, data, loss=None, opt=None, steps=1000):
    """ Generic optimization for free variables.
    Args:
        func: Function to optimize
        vars: List of variables to optimize
        data: Data dictionary
        loss: Loss function
        opt: Optimizer
        steps: Number of optimization steps
    Returns:
        Final loss value
    """
    if loss is None:
        loss = torch.nn.MSELoss()
    if opt is None:
        opt = torch.optim.Adam(vars)
    for _ in range(steps):
        opt.zero_grad()
        result = func(vars, data)
        l = loss(result, data["target"])
        l.backward()
        opt.step()
    return l.item()

# ---------------------------------------------------------------------------- #
# PNM export

def pnm(fd, tn):
    """ Export tensor to PGM/PBM format.
    Args:
        fd: File descriptor
        tn: Tensor to export
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
