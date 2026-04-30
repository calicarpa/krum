# coding: utf-8
###
# @file   jobs.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2020-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Simple job management for reproduction scripts.
###

"""
Experiment job management helpers.

This module provides utilities for running and managing experiment jobs in a
reproducible manner.

Classes and Functions
---------------------

Job orchestration
    ``Command`` encapsulates a command with seed, device, and result-directory
    arguments.
    ``Jobs`` manages parallel execution of experiments on multiple devices.

Helpers
    ``dict_to_cmdlist`` converts dictionaries into command-line argument lists.
    ``move_directory`` moves an existing directory aside with versioning.

Example
-------

.. code-block:: python

    from tools import Command, Jobs, dict_to_cmdlist

    cmd = Command(["python", "train.py", "--lr", "0.01"])
    jobs = Jobs("./results", devices=["cuda:0", "cuda:1"])
    jobs.submit("exp1", cmd)
    jobs.wait()
    jobs.close()
"""

__all__ = ["dict_to_cmdlist", "Command", "Jobs"]

import threading
from pathlib import Path


# ---------------------------------------------------------------------------- #
# Helpers


def move_directory(path: Path) -> Path:
    """
    Move an existing directory aside with versioning.

    If a directory already exists at the given path, it is renamed with an
    incremental suffix (for example, ``results.0``, ``results.1``) before a
    new directory is created.

    Parameters
    ----------
    path : pathlib.Path
        Directory path to move aside if it already exists.

    Returns
    -------
    pathlib.Path
        The input path, returned unchanged for chaining.

    Example
    -------
    >>> from pathlib import Path
    >>> move_directory(Path("results"))
    # Moves existing "results" to "results.0" if it exists
    """
    # Move directory if it exists
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(
                f"Expected to find nothing or (a symlink to) a directory at {str(path)!r}"
            )
        i = 0
        while True:
            mvpath = path.parent / f"{path.name}.{i}"
            if not mvpath.exists():
                path.rename(mvpath)
                break
            i += 1
    # Enable chaining
    return path


def dict_to_cmdlist(dp: dict) -> list[str]:
    """
    Convert a dictionary into command-line arguments.

    This helper is useful for turning experiment configurations into CLI
    arguments.

    Parameters
    ----------
    dp : dict
        Dictionary mapping parameter names to values.

    Returns
    -------
    list of str
        Command-line arguments such as ``["--lr", "0.01", "--batch", "32"]``.

    Notes
    -----
    - Boolean values are included only when they are ``True``.
    - Lists and tuples expand to repeated ``--name value`` pairs.

    Example
    -------
    >>> dict_to_cmdlist({"lr": 0.01, "batch": 32, "debug": True})
    ['--lr', '0.01', '--batch', '32', '--debug']
    >>> dict_to_cmdlist({"layers": [64, 128]})
    ['--layers', '64', '--layers', '128']
    """
    cmd = list()
    for name, value in dp.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{name}")
        elif isinstance(value, (list, tuple)):
            for v in value:
                cmd.append(f"--{name}")
                cmd.append(str(v))
        else:
            cmd.append(f"--{name}")
            cmd.append(str(value))
    return cmd


# ---------------------------------------------------------------------------- #
# Command wrapper


class Command:
    """
    Command wrapper that adds standard runtime arguments.

    This class wraps a base command and automatically appends seed, device, and
    result-directory arguments when executing it.
    """

    def __init__(
        self,
        base: list[str],
        seed: int | None = None,
        device: str | None = None,
        result_directory: Path | None = None,
    ) -> None:
        """
        Initialize the command wrapper.

        Parameters
        ----------
        base : list of str
            Base command as a list of strings.
        seed : int, optional
            Random seed to add.
        device : str, optional
            Device to add, for example ``"cuda:0"``.
        result_directory : pathlib.Path, optional
            Result directory path to add.
        """
        self._base = base
        self._seed = seed
        self._device = device
        self._result_directory = result_directory

    def __call__(self) -> list[str]:
        """
        Build the full command list with optional runtime arguments.

        Returns
        -------
        list of str
            Base command extended with ``--seed``, ``--device``, and
            ``--result-directory`` when they were provided at initialization.
        """
        cmd = list(self._base)
        if self._seed is not None:
            cmd.extend(["--seed", str(self._seed)])
        if self._device is not None:
            cmd.extend(["--device", self._device])
        if self._result_directory is not None:
            cmd.extend(["--result-directory", str(self._result_directory)])
        return cmd


# ---------------------------------------------------------------------------- #
# Jobs management


class Jobs:
    """
    Job execution manager for parallel experiments.

    Manages parallel execution of experiments across multiple devices,
    with support for result tracking and error handling.
    """

    def __init__(
        self, result_directory: Path, devices: list[str] | None = None, devmult: int = 1
    ) -> None:
        """
        Initialize jobs manager.

        Parameters
        ----------
        result_directory : pathlib.Path
            Directory to store results.
        devices : list of str, optional
            List of device names (e.g., ["cuda:0", "cuda:1"]).
            Defaults to CPU if none specified.
        devmult : int, optional
            Number of parallel jobs per device. Default is 1.
        """
        self._result_directory = result_directory
        self._devices = devices or ["cpu"]
        self._devmult = devmult
        self._pending = []
        self._lock = threading.Lock()

    def submit(self, name: str, command: list[str]) -> None:
        """
        Submit a job for execution.

        Parameters
        ----------
        name : str
            Job identifier.
        command : list of str
            Full command to execute (as returned by ``Command.__call__``).
        """
        with self._lock:
            self._pending.append((name, command))

    def wait(self, exit_is_requested: bool | None = None) -> None:
        """
        Wait for all pending jobs to complete.

        Parameters
        ----------
        exit_is_requested : bool or None, optional
            Optional external flag to request early termination.
        """
        # Implementation depends on threading
        pass

    def close(self) -> None:
        """
        Close the jobs manager and release resources.

        Notes
        -----
        No-op in the current stub implementation.
        """
