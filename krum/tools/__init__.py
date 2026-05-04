###
# @file   __init__.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Bunch of useful tools, but each too small to have its own package.
###

"""
Core Utility Module for Krum.

This module provides the fundamental infrastructure utilities used throughout
Krum, including logging, error handling, and common operations.

Key Components
--------------

**Exceptions:**

- ``UserException``: Base exception for user-facing errors
- ``Context``: Thread-local context for colored logging

**Logging:**

- ``info()``, ``success()``, ``warning()``, ``error()``: Colored logging functions
- ``fatal()``: Print error and exit

**I/O:**

- ``ContextIOWrapper``: Wrapper for stdout/stderr with context prefixing

**Module Loading:**

- ``import_directory()``: Load all Python modules from a directory
- ``import_exported_symbols()``: Import symbols from a module

**Utilities:**

- ``parse_keyval()``: Parse key:value CLI arguments
- ``fullqual()``: Get fully qualified name of objects
- ``onetime()``: Thread-safe one-time flag
"""

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from .jobs import Command, Jobs, dict_to_cmdlist
from .misc import (
    ClassRegister,
    MethodCallReplicator,
    TimedContext,
    UnavailableException,
    deltatime_format,
    deltatime_point,
    fatal_unavailable,
    fullqual,
    get_loaded_dependencies,
    interactive,
    line_maximize,
    localtime,
    onetime,
    pairwise,
    parse_keyval,
)
from .pytorch import (
    AccumulatedTimedContext,
    WeightedMSELoss,
    compute_avg_dev_max,
    flatten,
    grad_of,
    grads_of,
    pnm,
    regression,
    relink,
    weighted_mse_loss,
)

# ---------------------------------------------------------------------------- #
# User exception base class, print string representation and exit(1) on uncaught


class UserException(Exception):
    """
    Base exception for user-facing errors.
    """

    pass


# ---------------------------------------------------------------------------- #
# Context and color management


class Context:
    """
    Per-thread logging context and color manager.
    """

    # Constants
    __colors = {
        "header": "\033[1;30m",
        "red": "\033[1;31m",
        "error": "\033[1;31m",
        "green": "\033[1;32m",
        "success": "\033[1;32m",
        "yellow": "\033[1;33m",
        "warning": "\033[1;33m",
        "blue": "\033[1;34m",
        "info": "\033[1;34m",
        "gray": "\033[1;30m",
        "trace": "\033[1;30m",
    }
    __clrend = "\033[0m"

    # Thread-local variables
    __local = threading.local()

    @classmethod
    def __local_init(self):
        """
        Initialize thread-local context state if necessary.
        """
        if not hasattr(self.__local, "stack"):
            self.__local.stack = []  # List of pairs (context name, color code)
            self.__local.header = ""  # Current header string
            self.__local.color = self.__clrend  # Current color code

    @classmethod
    def __rebuild(self):
        """
        Rebuild the current log header and color from the context stack.
        """
        # Collect current header and color
        header = ""
        color = None
        for ctx, clr in reversed(self.__local.stack):
            if ctx is not None:
                header = "[" + ctx + "] " + header
            if clr is not None and color is None:
                color = clr
        if color is None:
            color = self.__clrend
        # Prepend thread name if not main thread
        cthrd = threading.current_thread()
        if cthrd != threading.main_thread():
            header = "[" + cthrd.name + "] " + header
        # Store the new header and color
        self.__local.header = header
        self.__local.color = color

    @classmethod
    def _get(self):
        """
        Return the current thread-local header and color escape sequences.

        Returns
        -------
        tuple[str, str, str, str]
            Current header, header color prefix, message color prefix, and color
            reset suffix.
        """
        self.__local_init()
        return (
            self.__local.header,
            self.__colors["header"],
            self.__local.color,
            self.__clrend,
        )

    def __init__(self, cntxtname: str | None, colorname: str | None) -> None:
        """
        Create a context stack entry.

        Parameters
        ----------
        cntxtname : str or None
            Context name to prepend to log lines, or ``None`` for no additional
            context.
        colorname : str or None
            Color name to apply while the context is active, or ``None`` to keep the
            current color.
        """
        # Color code resolution
        if colorname is None:
            colorcode = None
        else:
            assert colorname in type(self).__colors, "Unknown color name " + repr(colorname)
            colorcode = type(self).__colors[colorname]
        # Finalization
        self.__pair = (cntxtname, colorcode)

    def __enter__(self):
        """
        Enter the logging context.

        Returns
        -------
        Context
            This context manager instance.
        """
        type(self).__local_init()
        type(self).__local.stack.append(self.__pair)
        type(self).__rebuild()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """
        Leave the logging context.

        Parameters
        ----------
        *args : object
            Ignored positional arguments supplied by the context manager protocol.
        **kwargs : object
            Ignored keyword arguments supplied by the context manager protocol.
        """
        type(self).__local.stack.pop()
        type(self).__rebuild()


class ContextIOWrapper:
    """
    Context-aware text I/O wrapper.
    """

    def __init__(self, output: TextIO, nocolor: bool | None = None) -> None:
        """
        Wrap a text output stream.

        Parameters
        ----------
        output : object
            Wrapped stream-like object.
        nocolor : bool or None, optional
            Whether to disable ANSI colors. If ``None``, colors are disabled for
            non-TTY streams.
        """
        # Check whether to apply coloring if unset
        if nocolor is None:
            nocolor = not output.isatty()
        # Finalization
        self.__newline = True  # At a new line
        self.__colored = True  # Color has been applied
        self.__output = output
        self.__nocolor = nocolor

    def __getattr__(self, name: str) -> object:
        """
        Forward non-overridden attribute access to the wrapped stream.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute value from the wrapped stream.
        """
        return getattr(self.__output, name)

    def write(self, text: str) -> int:
        """
        Write text with the active context prefix and color.

        Parameters
        ----------
        text : str
            Text to write.

        Returns
        -------
        int
            Return value forwarded from the wrapped stream's ``write`` method.
        """
        # Get the current context
        header, clrheader, clrbegin, clrend = Context._get()
        if self.__nocolor:
            clrheader = ""
            clrbegin = ""
            clrend = ""
        # Prepend the header to every line
        lines = text.splitlines(True)
        text = ""
        for line in lines:
            if self.__newline:
                text += clrheader + header
            text += clrbegin
            self.__newline = True
            text += line
        if len(lines) > 0 and lines[-1][-len(os.linesep) :] != os.linesep:
            self.__newline = False
        # Write the modified text with the right color
        return self.__output.write(text + clrend)


def _make_color_print(color: str) -> Callable[..., object]:
    """
    Build a ``print`` wrapper that runs inside a colored context.

    Parameters
    ----------
    color : str
        Target color name.

    Returns
    -------
    object
        Print wrapper closure.
    """

    def color_print(*args, context: str | None = None, **kwargs) -> object:
        """
        Print inside the configured colored context.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to :func:`print`.
        context : str or None, optional
            Context name to use while printing.
        **kwargs : object
            Keyword arguments forwarded to :func:`print`.

        Returns
        -------
        object
            Return value forwarded from :func:`print`.
        """
        with Context(context, color):
            return print(*args, **kwargs)

    return color_print


# Explicit colored print shortcuts (required for static type checkers)
trace = _make_color_print("trace")
info = _make_color_print("info")
success = _make_color_print("success")
warning = _make_color_print("warning")
error = _make_color_print("error")


def fatal(*args, with_traceback: bool = False, **kwargs) -> None:
    """
    Print an error message and terminate the process with exit code 1.

    Parameters
    ----------
    *args : object
        Positional arguments forwarded to :func:`error`.
    with_traceback : bool, optional
        Whether to include the current traceback after the message.
    **kwargs : object
        Keyword arguments forwarded to :func:`error`.
    """
    global error
    error(*args, **kwargs)
    if with_traceback:
        with Context("traceback", "trace"):
            traceback.print_exc()
    exit(1)


# Wrap the standard text output wrappers
sys.stdout = ContextIOWrapper(sys.stdout)
sys.stderr = ContextIOWrapper(sys.stderr)

# ---------------------------------------------------------------------------- #
# Uncaught exception context wrapping


def uncaught_wrap(hook: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap an uncaught exception hook with contextual logging.

    Parameters
    ----------
    hook : object
        Uncaught exception hook to wrap.

    Returns
    -------
    object
        Wrapped uncaught exception hook.
    """

    def uncaught_call(etype: type, evalue: object, traceback: object) -> object:
        """
        Handle uncaught exceptions with user-facing context.

        Parameters
        ----------
        etype : type
            Exception class.
        evalue : object
            Exception value.
        traceback : object
            Traceback associated with the exception.

        Returns
        -------
        object
            Return value forwarded from the wrapped hook for non-user exceptions.
        """
        if issubclass(etype, UserException):
            with Context("fatal", "error"):
                print(evalue)
        else:
            with Context("uncaught", "error"):
                return hook(etype, evalue, traceback)
        return None

    return uncaught_call


# Wrap the original exception hook
sys.excepthook = uncaught_wrap(sys.excepthook)

# ---------------------------------------------------------------------------- #
# Local module loading and post-processing

_imported = {}  # Map symbol name -> module source name


def import_exported_symbols(name: str, module, scope: dict) -> None:
    """
    Import a module's exported symbols into a target scope.

    Parameters
    ----------
    name : str
        Source module name.
    module : module
        Loaded module instance.
    scope : dict
        Target scope to update with exported symbols.
    """
    global _imported
    if hasattr(module, "__all__"):
        for symname in module.__all__:
            # Check name
            if not hasattr(module, symname):
                with Context(None, "warning"):
                    print("Symbol " + repr(symname) + " exported but not defined")
                continue
            if symname in _imported:
                with Context(None, "warning"):
                    print("Symbol " + repr(symname) + " already exported by " + repr(_imported[symname]))
                continue
            if symname in scope:
                with Context(None, "warning"):
                    print("Symbol " + repr(symname) + " already exported by '__init__.py'")
                continue
            # Import in module scope
            scope[symname] = getattr(module, symname)
            _imported[symname] = name


def import_directory(
    dirpath: Path,
    scope: dict,
    post: Callable[..., Any] | None = import_exported_symbols,
    ignore: list[str] | None = None,
) -> None:
    """
    Import every Python module from a directory into a target scope.

    Parameters
    ----------
    dirpath : pathlib.Path
        Directory containing modules to import.
    scope : dict
        Target scope used for imports and post-processing.
    post : object, optional
        Post-import callback with signature ``(name, module, scope) -> None``.
    ignore : list[str], optional
        Module names to ignore.
    """
    # Import in the scope of the caller
    if ignore is None:
        ignore = ["__init__"]
    for path in dirpath.iterdir():
        if path.is_file() and path.suffix == ".py":
            name = path.stem
            if "." in name or name in ignore:
                continue
            with Context(name, None):
                try:
                    # Load module
                    base = __import__(scope["__package__"], scope, scope, [name], 0)
                    # Post processing
                    if callable(post):
                        post(name, getattr(base, name), scope)
                except Exception as err:
                    with Context(None, "warning"):
                        print("Loading failed for module " + repr(path.name) + ": " + str(err))
                        with Context("traceback", "trace"):
                            traceback.print_exc()


# Public API of the tools package
__all__ = [
    # Logging & context
    "Context",
    "ContextIOWrapper",
    "UserException",
    "trace",
    "info",
    "success",
    "warning",
    "error",
    "fatal",
    "uncaught_wrap",
    # Module loading
    "import_exported_symbols",
    "import_directory",
    # misc
    "ClassRegister",
    "MethodCallReplicator",
    "TimedContext",
    "UnavailableException",
    "deltatime_format",
    "deltatime_point",
    "fatal_unavailable",
    "fullqual",
    "get_loaded_dependencies",
    "interactive",
    "line_maximize",
    "localtime",
    "onetime",
    "pairwise",
    "parse_keyval",
    # pytorch
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
    # jobs
    "Command",
    "Jobs",
    "dict_to_cmdlist",
]
