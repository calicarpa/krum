# coding: utf-8
###
# @file   optimizer.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Optimizer wrapper.
###

"""
Optimizer wrapper that resolves PyTorch optimizers by name.

This module provides a thin wrapper around ``torch.optim`` that allows
optimizers to be instantiated from CLI strings while exposing a uniform
interface for learning-rate adjustments.

Example
-------

>>> from experiments import Optimizer, Model
>>> model = Model("lenet", num_classes=10)
>>> optim = Optimizer("adam", model, lr=0.001)
>>> optim.set_lr(0.0001)
"""

__all__ = ["Optimizer"]

import tools

import torch

# ---------------------------------------------------------------------------- #
# Optimizer wrapper class


class Optimizer:
    """
    Optimizer wrapper with name resolution and LR control.

    Parameters
    ----------
    name_build : str or callable
        Optimizer name (e.g. ``"adam"``, ``"sgd"``) or a constructor
        function. Names are resolved against ``torch.optim``.
    model : experiments.Model
        Model whose parameters will be optimized.
    *args : object
        Additional positional arguments forwarded to the optimizer
        constructor.
    **kwargs : object
        Additional keyword arguments forwarded to the optimizer
        constructor.

    Raises
    ------
    tools.UnavailableException
        If ``name_build`` is a string that does not match any known
        optimizer.
    """

    # Map 'lower-case names' -> 'optimizer constructor' available in PyTorch
    __optimizers = None

    @classmethod
    def _get_optimizers(cls):
        """
        Lazily build the name-to-class mapping for PyTorch optimizers.

        Returns
        -------
        dict[str, type]
            Mapping from lower-case names to optimizer classes.
        """
        # Fast-path already loaded
        if cls.__optimizers is not None:
            return cls.__optimizers
        # Initialize the dictionary
        cls.__optimizers = dict()
        # Populate with TorchVision optimizers
        for name in dir(torch.optim):
            if len(name) == 0 or name[0] == "_":
                continue
            builder = getattr(torch.optim, name)
            if (
                isinstance(builder, type)
                and builder is not torch.optim.Optimizer
                and issubclass(builder, torch.optim.Optimizer)
            ):
                cls.__optimizers[name.lower()] = builder
        return cls.__optimizers

    def __init__(self, name_build, model, *args, **kwargs):
        """
        Initialize the optimizer wrapper.

        Parameters
        ----------
        name_build : str or callable
            Optimizer name or constructor function.
        model : experiments.Model
            Model to optimize.
        *args : object
            Forwarded to the optimizer constructor.
        **kwargs : object
            Forwarded to the optimizer constructor.
        """
        # Recover name/constructor
        if callable(name_build):
            name = tools.fullqual(name_build)
            build = name_build
        else:
            optims = type(self)._get_optimizers()
            name = str(name_build)
            build = optims.get(name, None)
            if build is None:
                raise tools.UnavailableException(optims, name, what="optimizer name")
        # Build optimizer
        optim = build(model._model.parameters(), *args, **kwargs)
        # Finalization
        self._optim = optim
        self._name = name

    def __getattr__(self, *args):
        """
        Forward attribute access to the wrapped optimizer.

        Parameters
        ----------
        *args : object
            Either ``(name,)`` or ``(name, default)``.

        Returns
        -------
        object
            Attribute from the wrapped optimizer instance.

        Raises
        ------
        RuntimeError
            If called with more than two positional arguments.
        """
        if len(args) == 1:
            return getattr(self._optim, args[0])
        if len(args) == 2:
            return getattr(self._optim, args[0], args[1])
        raise RuntimeError(
            "'Optimizer.__getattr__' called with the wrong number of parameters"
        )

    def __str__(self):
        """
        Return a printable representation.

        Returns
        -------
        str
            Human-readable optimizer name.
        """
        return f"optimizer {self._name}"

    def set_lr(self, lr):
        """
        Set the learning rate for all parameter groups.

        Parameters
        ----------
        lr : float
            New learning rate.
        """
        for pg in self._optim.param_groups:
            pg["lr"] = lr
