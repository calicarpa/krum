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

"""Experiment job management helpers.

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

Example:
-------

.. code-block:: python

    from tools import Command, Jobs, dict_to_cmdlist

    cmd = Command(["python", "train.py", "--lr", "0.01"])
    jobs = Jobs("./results", devices=["cuda:0", "cuda:1"])
    jobs.submit("exp1", cmd)
    jobs.wait()
    jobs.close()
"""

__all__ = ["Command", "Jobs", "dict_to_cmdlist"]

import shlex
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from krum import tools

# ---------------------------------------------------------------------------- #
# Helpers


def move_directory(path: Path) -> Path:
    """Move an existing directory aside with versioning.

    If a directory already exists at the given path, it is renamed with an
    incremental suffix (for example, ``results.0``, ``results.1``) before a
    new directory is created.

    Parameters
    ----------
    path : pathlib.Path
        Directory path to move aside if it already exists.

    Returns:
    -------
    pathlib.Path
        The input path, returned unchanged for chaining.

    Raises:
    ------
    RuntimeError
        If ``path`` exists but is not a directory (or a symlink to one).

    Example:
    -------
    >>> from pathlib import Path
    >>> move_directory(Path("results"))
    # Moves existing "results" to "results.0" if it exists
    """
    # Move directory if it exists
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Expected to find nothing or (a symlink to) a directory at {str(path)!r}")
        i = 0
        while True:
            mvpath = path.parent / f"{path.name}.{i}"
            if not mvpath.exists():
                path.rename(mvpath)
                break
            i += 1
    # Enable chaining
    return path


def dict_to_cmdlist(dp: dict[str, Any]) -> list[str]:
    """Convert a dictionary into command-line arguments.

    This helper is useful for turning experiment configurations into CLI
    arguments.

    Parameters
    ----------
    dp : dict of str to Any
        Dictionary mapping parameter names to values.

    Returns:
    -------
    list of str
        Command-line arguments such as ``["--lr", "0.01", "--batch", "32"]``.

    Notes:
    -----
    - Boolean values are included only when they are ``True``.
    - Lists and tuples expand to repeated ``--name value`` pairs.

    Example:
    -------
    >>> dict_to_cmdlist({"lr": 0.01, "batch": 32, "debug": True})
    ['--lr', '0.01', '--batch', '32', '--debug']
    >>> dict_to_cmdlist({"layers": [64, 128]})
    ['--layers', '64', '--layers', '128']
    """
    cmd: list[str] = []
    for name, value in dp.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{name}")
        elif isinstance(value, (list, tuple)):
            cmd.append(f"--{name}")
            for subval in value:
                cmd.append(str(subval))
        elif value is not None:
            cmd.append(f"--{name}")
            cmd.append(str(value))
    return cmd


# ---------------------------------------------------------------------------- #
# Job command class


class Command:
    """Command wrapper that adds standard runtime arguments.

    This class wraps a base command and automatically appends seed, device, and
    result-directory arguments when building the final command line.

    Parameters
    ----------
    command : iterable of str
        Base command as an iterable of strings (e.g. ``["python", "train.py"]``).
        The iterable is copied on instantiation.

    Attributes:
    ----------
    _basecmd : list of str
        Internal copy of the base command.
    """

    def __init__(self, command: Iterable[str]) -> None:
        """Initialize the command builder.

        Parameters
        ----------
        command : iterable of str
            Base command as an iterable of strings.
        """
        self._basecmd: list[str] = list(command)

    def build(self, seed: int | str, device: str, resdir: Path | str) -> list[str]:
        """Build the final command line.

        Parameters
        ----------
        seed : int or str
            Seed to use for the experiment.
        device : str
            Device on which to run the experiment (e.g. ``"cuda:0"``).
        resdir : pathlib.Path or str
            Target directory path for results.

        Returns:
        -------
        list of str
            Final command list ready to be passed to ``subprocess.run``.
        """
        cmd = self._basecmd.copy()
        for name, value in (
            ("seed", seed),
            ("device", device),
            ("result-directory", resdir),
        ):
            cmd.append(f"--{name}")
            cmd.append(shlex.quote(value if isinstance(value, str) else str(value)))
        return cmd


# ---------------------------------------------------------------------------- #
# Job class


class Jobs:
    """Job execution manager for parallel experiments.

    Manages parallel execution of experiments across multiple devices,
    with support for result tracking and error handling.

    Parameters
    ----------
    res_dir : pathlib.Path or str
        Directory to store results.
    devices : list of str, optional
        List of device names (e.g. ``["cuda:0", "cuda:1"]``).
        Defaults to ``["cpu"]`` if none specified.
    devmult : int, optional
        Number of parallel jobs per device. Default is ``1``.
    seeds : sequence of int, optional
        Seeds to use for repeating experiments. Default is ``range(1, 6)``.

    Attributes:
    ----------
    _res_dir : pathlib.Path
        Resolved result directory.
    _jobs : list of tuple or None
        Pending job queue as ``(name, seed, command)`` tuples, or ``None``
        when the manager has been closed.
    _workers : list of threading.Thread
        Worker thread pool, one entry per active slot.
    _devices : list of str
        Devices used for execution.
    _seeds : tuple of int
        Seeds used for repeating experiments.
    _lock : threading.Lock
        Main lock protecting shared state.
    _cvready : threading.Condition
        Condition variable to signal that new jobs are available or that
        workers must shut down.
    _cvdone : threading.Condition
        Condition variable to signal that all submitted jobs have been
        processed.
    """

    @staticmethod
    def _run(
        topdir: Path,
        name: str,
        seed: int,
        device: str,
        command: Command,
    ) -> None:
        """Run a single experiment with the given parameters.

        Parameters
        ----------
        topdir : pathlib.Path
            Parent result directory.
        name : str
            Experiment unique name.
        seed : int
            Experiment seed.
        device : str
            Device on which to run the experiment.
        command : Command
            Command builder to use.
        """
        # Add seed to name
        name = f"{name}-{seed}"
        # Process experiment
        with tools.Context(name, "info"):
            finaldir = topdir / name
            # Check whether the experiment was already successful
            if finaldir.exists():
                tools.info("Experiment already processed.")
                return
            # Move-make the pending result directory
            resdir = move_directory(topdir / f"{name}.pending")
            resdir.mkdir(mode=0o755, parents=True)
            # Build the command
            args = command.build(seed, device, resdir)
            # Launch the experiment and capture the standard output/error
            tools.trace(" ".join(shlex.quote(arg) for arg in args))
            cmd_res = subprocess.run(args, check=False, capture_output=True)
            if cmd_res.returncode == 0:
                tools.info("Experiment successful")
            else:
                tools.warning("Experiment failed")
                finaldir = topdir / f"{name}.failed"
                move_directory(finaldir)
            resdir.rename(finaldir)
            (finaldir / "stdout.log").write_bytes(cmd_res.stdout)
            (finaldir / "stderr.log").write_bytes(cmd_res.stderr)

    def _worker_entrypoint(self, device: str) -> None:
        """Worker thread entry point.

        Continuously picks pending jobs from the queue and executes them on
        the assigned device until the manager is closed.

        Parameters
        ----------
        device : str
            Device assigned to this worker.
        """
        while True:
            # Take a pending experiment, or exit if requested
            with self._lock:
                while True:
                    # Check if must exit
                    if self._jobs is None:
                        return
                    # Check and pick the first pending experiment, if available
                    if len(self._jobs) > 0:
                        name, seed, command = self._jobs.pop()
                        break
                    # Wait for new job notification
                    self._cvready.wait()
            # Run the picked experiment
            self._run(self._res_dir, name, seed, device, command)

    def __init__(
        self,
        res_dir: Path | str,
        devices: list[str] | None = None,
        devmult: int = 1,
        seeds: Sequence[int] | None = None,
    ) -> None:
        """Initialize the experiment launcher.

        Parameters
        ----------
        res_dir : pathlib.Path or str
            Target directory path for results.
        devices : list of str, optional
            Devices on which to run experiments (e.g. ``["cuda:0"]``).
            Defaults to ``["cpu"]``.
        devmult : int, optional
            Device multiplier. Defaults to 1.
        seeds : sequence of int, optional
            Seeds to use for the experiments. Defaults to ``range(1, 6)``.
        """
        # Initialize instance
        if devices is None:
            devices = ["cpu"]
        if seeds is None:
            seeds = tuple(range(1, 6))
        self._res_dir: Path = Path(res_dir)
        self._jobs: list[tuple[str, int, Command]] | None = []
        self._workers: list[threading.Thread] = []
        self._devices: list[str] = devices
        self._seeds: tuple[int, ...] = tuple(seeds)
        self._lock = threading.Lock()
        self._cvready = threading.Condition(
            lock=self._lock
        )  # Signal jobs have been added and must be processed, or the worker must quit
        self._cvdone = threading.Condition(lock=self._lock)  # Signal jobs have all been processed
        # Launch the worker pool
        for _ in range(devmult):
            for device in devices:
                thread = threading.Thread(target=self._worker_entrypoint, name=device, args=(device,))
                thread.start()
                self._workers.append(thread)

    def get_seeds(self) -> tuple[int, ...]:
        """Get the list of seeds used for repeating the experiments.

        Returns:
        -------
        tuple of int
            Seeds used by this manager.
        """
        return self._seeds

    def close(self) -> None:
        """Close and wait for the worker pool, discarding not-yet-started submissions."""
        # Close the manager
        with self._lock:
            # Check if already closed
            if self._jobs is None:
                return
            # Reset submission list
            self._jobs = None
            # Notify all the workers
            self._cvready.notify_all()
        # Wait for all the workers
        for worker in self._workers:
            worker.join()

    def submit(self, name: str, command: Command) -> None:
        """Submit a job for execution.

        The job is repeated for every seed configured in the manager.

        Parameters
        ----------
        name : str
            Job identifier.
        command : Command
            Command builder to execute.

        Raises:
        ------
        RuntimeError
            If the manager has already been closed.
        """
        with self._lock:
            # Check if not closed
            if self._jobs is None:
                raise RuntimeError("Experiment manager cannot take new jobs as it has been closed")
            # Submit the experiment with each seed
            for seed in self._seeds:
                self._jobs.insert(0, (name, seed, command))
            self._cvready.notify(n=len(self._seeds))

    def wait(self, predicate: Callable[[], bool] | None = None) -> None:
        """Wait for all the submitted jobs to be processed.

        Parameters
        ----------
        predicate : callable returning bool, optional
            Optional custom predicate. If provided, waiting stops when the
            predicate returns ``True``.
        """
        while True:
            with self._lock:
                # Wait for condition or timeout
                self._cvdone.wait(timeout=1.0)
                # Check status
                if self._jobs is None:
                    break
                if len(self._jobs) == 0:
                    break
                if not any(worker.is_alive() for worker in self._workers):
                    break
                if predicate is not None and predicate():
                    break
