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
This module provides utilities for running and managing experiment jobs
in a reproducible manner.

Classes and Functions
---------------------

**Job Orchestration:**

- ``Command``: Encapsulates a command with seed, device, and result directory
- ``Jobs``: Manages parallel execution of experiments on multiple devices

**Helpers:**

- ``dict_to_cmdlist``: Convert dictionary to command-line argument list
- ``move_directory``: Move existing directory with versioning

Example
-------

.. code-block:: python

    from tools import Command, Jobs, dict_to_cmdlist

    # Create command
    cmd = Command(["python", "train.py", "--lr", "0.01"])

    # Run jobs
    jobs = Jobs("./results", devices=["cuda:0", "cuda:1"])
    jobs.submit("exp1", cmd)
    jobs.wait()
    jobs.close()
"""

__all__ = ["dict_to_cmdlist", "Command", "Jobs"]

import shlex
import subprocess
import threading
from pathlib import Path

import tools

# ---------------------------------------------------------------------------- #
# Helpers


def move_directory(path: Path) -> Path:
    """
    Move existing directory to a new location with versioning.

    If a directory already exists at the given path, it is renamed with
    an incremental suffix (e.g., "results.0", "results.1") before creating
    a new directory.

    Parameters
    ----------
    path : pathlib.Path
        Path to the directory to create.

    Returns
    -------
    pathlib.Path
        The input path (for chaining).

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
    Transform a dictionary into a list of command-line arguments.

    This is useful for converting experiment configurations into CLI commands.

    Parameters
    ----------
    dp : dict
        Dictionary mapping parameter names to values.

    Returns
    -------
    list of str
        Command-line arguments (e.g., ["--lr", "0.01", "--batch", "32"]).

    Notes
    -----
    - For boolean values: parameter is included only if True
    - For lists/tuples: parameter is followed by each value

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
    Command wrapper that adds standard arguments.

    Wraps a base command and automatically adds seed, device, and result
    directory arguments when executing.
    """

    def __init__(self, base: list[str], seed: int | None = None, device: str | None = None, result_directory: Path | None = None) -> None:
        """
        Initialize command wrapper.

        Parameters
        ----------
        base : list of str
            Base command as list of strings.
        seed : int, optional
            Random seed to add.
        device : str, optional
            Device to add (e.g., "cuda:0").
        result_directory : str, optional
            Result directory path to add.
        """
        self._base = base
        self._seed = seed
        self._device = device
        self._result_directory = result_directory

    def __call__(self):
        """Get the full command as list."""
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

    def __init__(self, result_directory: Path, devices: list[str] | None = None, devmult: int = 1) -> None:
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
        """ Submit a job for execution. """
        with self._lock:
            self._pending.append((name, command))

    def wait(self, exit_is_requested: bool | None = None) -> None:
        """ Wait for all pending jobs to complete. """
        # Implementation depends on threading
        pass

    def close(self) -> None:
        """ Close the jobs manager. """
        pass
