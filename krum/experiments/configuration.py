###
# @file   configuration.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Configuration wrapper.
###

"""
Tensor configuration wrapper.

This module provides the :class:`Configuration` class, an immutable mapping
that bundles ``device``, ``dtype``, and memory-transfer options. It is used
throughout ``experiments`` to ensure every created or moved tensor uses the
same configuration.

Example
-------

.. code-block:: python

    from experiments.configuration import Configuration

    config = Configuration(device="cuda:0", dtype=torch.float32)
    config["device"]  # device(type='cuda', index=0)
"""

__all__ = ["Configuration"]

from collections.abc import Mapping

from .. import tools
import torch

# ---------------------------------------------------------------------------- #
# Trivial tensor configuration holder (dtype, device, ...) class


class Configuration(Mapping):
    """
    Immutable tensor configuration holder.

    This class bundles ``device``, ``dtype``, and memory-transfer options
    into a single immutable mapping. It is used throughout ``experiments``
    to ensure every created or moved tensor uses the same configuration.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Target device. ``None`` defaults to ``"cuda"`` when available,
        otherwise ``"cpu"``. Strings such as ``"cuda:0"`` are resolved
        automatically.
    dtype : torch.dtype or None, optional
        Tensor datatype. ``None`` uses PyTorch's current default dtype.
    noblock : bool, optional
        Whether to use non-blocking host-to-device transfers.
    relink : bool, optional
        Whether to relink instead of copying during parameter assignments.

    Example
    -------

    >>> from experiments import Configuration
    >>> config = Configuration(device="cpu", dtype=torch.float32)
    >>> config["device"]
    device(type='cpu')
    """

    # Default selected device (GPU if available, else CPU)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, device=None, dtype=None, noblock=False, relink=False):
        """
        Initialize the configuration.

        Parameters
        ----------
        device : str, torch.device, or None, optional
            Target device. ``None`` selects the default device.
        dtype : torch.dtype or None, optional
            Tensor datatype.
        noblock : bool, optional
            Use non-blocking transfers.
        relink : bool, optional
            Relink instead of copy in assignments.
        """
        # Convert formatted device name to device instance
        if device is None:
            # Use default device
            device = type(self).default_device
        if isinstance(device, str):
            # Warn if CUDA is requested but not available
            if not torch.cuda.is_available() and device[:4] == "cuda":
                device = "cpu"
                tools.warning(
                    "CUDA is unavailable on this node, falling back to CPU in the configuration",
                    context="experiments",
                )
            # Convert
            device = torch.device(device)
        # Resolve the current default dtype if unspecified
        if dtype is None:
            dtype = torch.get_default_dtype()
        # Finalization
        self._args = {"device": device, "dtype": dtype, "non_blocking": noblock}
        self.relink = relink

    def __len__(self):
        """
        Return the number of configuration entries.

        Returns
        -------
        int
            Number of entries in the configuration mapping.
        """
        return len(self._args)

    def __getitem__(self, name):
        """
        Get a configuration value by name.

        Parameters
        ----------
        name : str
            Configuration key (e.g. ``"device"``, ``"dtype"``).

        Returns
        -------
        object
            Associated configuration value.
        """
        return self._args[name]

    def __iter__(self):
        """
        Iterate over all configuration keys.

        Returns
        -------
        iterator
            Iterator over configuration entry names.
        """
        return self._args.__iter__()

    def __str__(self):
        """
        Return a nicely printable representation.

        Returns
        -------
        str
            Human-readable configuration summary.
        """
        temp = self._args.copy()
        temp["relink"] = self.relink
        return str(temp)

    def __repr__(self):
        """
        Return an evaluable string representation.

        Returns
        -------
        str
            Python-code string that evaluates to this configuration.
        """
        display = {"non_blocking": "noblock"}
        argrepr = (", ").join(
            f"{display.get(key, key)}={val!r}" for key, val in self._args.items()
        )
        return f"Configuration({argrepr}, relink={self.relink})"
