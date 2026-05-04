###
# @file   checkpoint.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Checkpoint helpers.
###

"""
Checkpoint management for model, optimizer, and arbitrary stateful objects.

This module provides :class:`Checkpoint` for saving and restoring state
dictionaries, and :class:`Storage` for plain-dictionary checkpointing.

Example
-------

>>> from experiments import Checkpoint, Model, Optimizer
>>> ckpt = Checkpoint()
>>> ckpt.snapshot(model).snapshot(optimizer)
>>> ckpt.save("run.pt")
>>> # Later...
>>> ckpt.load("run.pt")
>>> ckpt.restore(model).restore(optimizer)
"""

__all__ = ["Checkpoint", "Storage"]

import copy
import pathlib

import tools
import torch

from .model import Model
from .optimizer import Optimizer

# ---------------------------------------------------------------------------- #
# Checkpoint helper class


class Checkpoint:
    """
    Collection of state dictionaries with saving/loading helpers.

    This class can snapshot any object implementing the ``state_dict`` /
    ``load_state_dict`` protocol (e.g. ``torch.nn.Module``,
    ``torch.optim.Optimizer``). It also knows how to unwrap
    :class:`~experiments.model.Model` and
    :class:`~experiments.optimizer.Optimizer` wrappers automatically.

    Example
    -------

    >>> ckpt = Checkpoint()
    >>> ckpt.snapshot(model, deepcopy=True)
    >>> ckpt.restore(model)
    """

    # Transfer for handling local package's classes
    _transfers = {
        Model: (lambda x: x._model),
        Optimizer: (lambda x: x._optim),
    }

    @classmethod
    def _prepare(cls, instance):
        """
        Prepare an instance for checkpointing.

        If the instance is a wrapped :class:`Model` or :class:`Optimizer`,
        the underlying PyTorch object is returned instead.

        Parameters
        ----------
        instance : object
            Instance to snapshot or restore.

        Returns
        -------
        tuple[object, str]
            Checkpoint-able instance and its fully-qualified storage key.

        Raises
        ------
        tools.UserException
            If the instance lacks ``state_dict`` or ``load_state_dict``.
        """
        # Recover instance's class
        inst_cls = type(instance)
        # Transfer if available
        if inst_cls in cls._transfers:
            res = cls._transfers[inst_cls](instance)
        else:
            res = instance
        # Assert the instance is checkpoint-able
        for prop in ("state_dict", "load_state_dict"):
            if not callable(getattr(res, prop, None)):
                raise tools.UserException(
                    f"Given instance {instance!r} is not checkpoint-able "
                    f"(missing callable member {prop!r})"
                )
        # Return the instance and the associated storage key
        return res, tools.fullqual(inst_cls)

    def __init__(self):
        """
        Create an empty checkpoint.
        """
        self._store = {}
        if __debug__:
            self._copied = {}

    def snapshot(self, instance, overwrite=False, deepcopy=False, nowarnref=False):
        """
        Take (or overwrite) a snapshot of an instance's state dictionary.

        Parameters
        ----------
        instance : object
            Instance to snapshot. Must support ``state_dict()``.
        overwrite : bool, optional
            Whether to overwrite an existing snapshot for the same class.
        deepcopy : bool, optional
            Whether to deep-copy the state dictionary instead of
            shallow-copying.
        nowarnref : bool, optional
            Suppress the debug warning when restoring a reference is the
            intended behavior.

        Returns
        -------
        Checkpoint
            Self, for chaining.

        Raises
        ------
        tools.UserException
            If a snapshot already exists and ``overwrite`` is ``False``.
        """
        instance, key = type(self)._prepare(instance)
        # Snapshot the state dictionary
        if not overwrite and key in self._store:
            raise tools.UserException(
                f"A snapshot for {key!r} is already stored in the checkpoint"
            )
        if deepcopy:
            self._store[key] = copy.deepcopy(instance.state_dict())
        else:
            self._store[key] = instance.state_dict().copy()
        # Track whether a deepcopy was made
        if __debug__:
            self._copied[key] = deepcopy or nowarnref
        # Enable chaining
        return self

    def restore(self, instance, nothrow=False):
        """
        Restore an instance from its stored snapshot.

        Parameters
        ----------
        instance : object
            Instance to restore. Must support ``load_state_dict()``.
        nothrow : bool, optional
            If ``True``, silently skip when no snapshot is available.

        Returns
        -------
        Checkpoint
            Self, for chaining.

        Raises
        ------
        tools.UserException
            If no snapshot exists and ``nothrow`` is ``False``.
        """
        instance, key = type(self)._prepare(instance)
        # Restore the state dictionary
        if key in self._store:
            instance.load_state_dict(self._store[key])
            # Check if restoring a reference
            if __debug__ and not self._copied.get(key, True):
                tools.warning(
                    f"Restoring a state dictionary reference in an instance of "
                    f"{tools.fullqual(type(instance))}; the resulting behavior "
                    f"may not be the one expected"
                )
        elif not nothrow:
            raise tools.UserException(
                f"No snapshot for {key!r} is available in the checkpoint"
            )
        # Enable chaining
        return self

    def load(self, filepath, overwrite=False):
        """
        Load checkpoint data from a file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the saved checkpoint.
        overwrite : bool, optional
            Whether to overwrite any existing snapshots.

        Returns
        -------
        Checkpoint
            Self, for chaining.

        Raises
        ------
        tools.UserException
            If the checkpoint is non-empty and ``overwrite`` is ``False``.
        """
        # Check if empty
        if not overwrite and len(self._store) > 0:
            raise tools.UserException("Unable to load into a non-empty checkpoint")
        # Load the file
        self._store = torch.load(filepath)
        # Reset the 'copied' flags accordingly
        if __debug__:
            self._copied.clear()
            for key in self._store.keys():
                self._copied[key] = True
        # Enable chaining
        return self

    def save(self, filepath, overwrite=False):
        """
        Save the current checkpoint to a file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination path.
        overwrite : bool, optional
            Whether to overwrite an existing file.

        Returns
        -------
        Checkpoint
            Self, for chaining.

        Raises
        ------
        tools.UserException
            If the file exists and ``overwrite`` is ``False``.
        """
        # Check if file already exists
        if pathlib.Path(filepath).exists() and not overwrite:
            raise tools.UserException(
                f"Unable to save checkpoint in existing file {str(filepath)!r} "
                f"(overwriting has not been allowed by the caller)"
            )
        # (Over)write the file
        torch.save(self._store, filepath)
        # Enable chaining
        return self


# ---------------------------------------------------------------------------- #
# Dictionary that implements "state_dict protocol"


class Storage(dict):
    """
    Plain dictionary that implements the ``state_dict`` protocol.

    This allows arbitrary key/value data to be snapshotted and restored
    alongside models and optimizers using :class:`Checkpoint`.
    """

    def state_dict(self):
        """
        Return the dictionary itself as state.

        Returns
        -------
        dict
            Self.
        """
        return self

    def load_state_dict(self, state):
        """
        Replace contents with the given state.

        Parameters
        ----------
        state : dict
            New dictionary contents.
        """
        self.update(state)
