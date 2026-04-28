# coding: utf-8
###
 # @file   misc.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Miscellaneous Python helpers.
###

"""
Utilities shared across the repository.

This module groups small helpers used for exception handling, parsing, timing,
interactive exploration, and light registry patterns.

Categories
----------

Exception handling
    ``UnavailableException`` and ``fatal_unavailable`` build consistent
    user-facing errors for missing registry entries.

Registry helpers
    ``MethodCallReplicator`` and ``ClassRegister`` provide small reusable
    patterns for dispatching calls and registering classes.

Parsing helpers
    ``parse_keyval`` and ``fullqual`` handle CLI-style key/value parsing and
    qualified-name formatting.

Timing helpers
    ``TimedContext``, ``onetime``, ``localtime``, ``deltatime_point``, and
    ``deltatime_format`` support timing and one-shot flags.

Miscellaneous helpers
    ``pairwise``, ``line_maximize``, ``interactive``, and
    ``get_loaded_dependencies`` cover assorted convenience tasks.

Example
-------

.. code-block:: python

    from tools import UnavailableException, parse_keyval, TimedContext

    try:
        raise UnavailableException({"a": 1, "b": 2}, "c", "option")
    except UnavailableException as e:
        print(e)

    args = parse_keyval(["lr:0.01", "batch:32"])
    with TimedContext("my_operation"):
        pass
"""

__all__ = [
    "UnavailableException",
    "fatal_unavailable",
    "MethodCallReplicator",
    "ClassRegister",
    "parse_keyval",
    "fullqual",
    "onetime",
    "TimedContext",
    "interactive",
    "get_loaded_dependencies",
    "line_maximize",
    "pairwise",
    "localtime",
    "deltatime_point",
    "deltatime_format",
]

import os
import pathlib
import site
import sys
import threading
import time
import traceback

import tools

# ---------------------------------------------------------------------------- #
# Unavailable user exception class


def make_unavailable_exception_text(
    data: list[str], name: str, what: str = "entry"
) -> str:
    """
    Build the message used by :class:`UnavailableException`.

    Parameters
    ----------
    data : list[str]
        Available names that the user could have selected.
    name : str
        Requested name that was not found in ``data``.
    what : str, optional
        Human-readable description of the named objects.

    Returns
    -------
    str
        User-facing message that lists the available names, or states that no
        names are available.
    """
    # Preparation
    if len(data) == 0:
        end = "no %s available" % what
    else:
        sep = "%s· " % os.linesep
        end = "expected one of:%s%s" % (sep, sep.join(data))
    # Final string cat
    return "Unknown %s %r, %s" % (what, name, end)


def fatal_unavailable(*args, **kwargs) -> None:
    """
    Report an unavailable entry as a fatal user-facing error.

    Parameters
    ----------
    *args : str
        Positional arguments forwarded to
        :func:`make_unavailable_exception_text`.
    **kwargs : str
        Keyword arguments forwarded to
        :func:`make_unavailable_exception_text`.
    """
    tools.fatal(make_unavailable_exception_text(*args, **kwargs))


class UnavailableException(tools.UserException):
    """User-facing exception raised when a selected registry entry is missing."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the exception message.

        Parameters
        ----------
        *args : str
            Positional arguments forwarded to
            :func:`make_unavailable_exception_text`.
        **kwargs : str
            Keyword arguments forwarded to
            :func:`make_unavailable_exception_text`.
        """
        # Finalization
        self._text = make_unavailable_exception_text(*args, **kwargs)

    def __str__(self):
        """Return the formatted explanatory message."""
        return self._text


# ---------------------------------------------------------------------------- #
# Simple method call replicator


class MethodCallReplicator:
    """Proxy that replicates method calls across multiple instances.

    Accessing an attribute returns a callable that invokes the same named
    attribute on each bound instance, in order, and returns the list of results.
    """

    def __init__(self, *args: object) -> None:
        """
        Bind the instances that should receive replicated method calls.

        Parameters
        ----------
        *args : object
            Instances on which to replicate method calls, in call order.
        """
        # Assertions
        assert len(args) > 0, (
            "Expected at least one instance on which to forward method calls"
        )
        # Finalization
        self.__instances = args

    def __getattr__(self, name: str) -> object:
        """
        Return a closure that replicates the named method call.

        Parameters
        ----------
        name : str
            Name of the method or callable attribute to replicate.

        Returns
        -------
        object
            Callable that forwards its arguments to every target callable and
            returns their results as a list.
        """
        # Target closures
        closures = [getattr(instance, name) for instance in self.__instances]

        # Replication closure
        def calls(*args, **kwargs) -> list[object]:
            """
            Call each target callable with the provided arguments.

            Parameters
            ----------
            *args : object
                Positional arguments forwarded to every target callable.
            **kwargs : object
                Keyword arguments forwarded to every target callable.

            Returns
            -------
            list[object]
                Results returned by the target callables, in instance order.
            """
            return [closure(*args, **kwargs) for closure in closures]

        # Build the replication closure
        return calls


# ---------------------------------------------------------------------------- #
# Simple class register


class ClassRegister:
    """Minimal registry mapping user-visible names to classes."""

    def __init__(self, singular: str, optplural: str | None = None) -> None:
        """
        Create an empty class registry.

        Parameters
        ----------
        singular : str
            Singular description of a registered class, used in error messages.
        optplural : str | None, optional
            Optional plural description, e.g. ``"class(es)"`` for ``"class"``.
            Defaults to ``singular + "(s)"``.
        """
        # Value deduction
        if optplural is None:
            optplural = singular + "(s)"
        # Finalization
        self.__denoms = (singular, optplural)
        self.__register = {}

    def itemize(self) -> list[str]:
        """Return the registered class names."""
        return self.__register.keys()

    def register(self, name: str, cls: type) -> None:
        """
        Register a class under a user-visible name.

        Parameters
        ----------
        name : str
            Name used to retrieve the class.
        cls : type
            Class associated with ``name``.
        """
        # Assertions
        assert name not in self.__register, (
            "Name "
            + repr(name)
            + " already in use while registering "
            + repr(
                getattr(
                    cls, "__name__", "<unknown " + self.__denoms[0] + " class name>"
                )
            )
        )
        # Registering
        self.__register[name] = cls

    def instantiate(self, name: str, *args, **kwargs) -> object:
        """
        Instantiate the class registered under ``name``.

        Parameters
        ----------
        name : str
            Registered class name.
        *args : object
            Positional arguments forwarded to the class constructor.
        **kwargs : object
            Keyword arguments forwarded to the class constructor.

        Returns
        -------
        object
            Instance of the registered class.

        Raises
        ------
        tools.UserException
            If ``name`` is not registered.
        """
        # Assertions
        if name not in self.__register:
            cause = "Unknown name " + repr(name) + ", "
            if len(self.__register) == 0:
                cause += "no registered " + self.__denoms[0]
            else:
                cause += (
                    "available "
                    + self.__denoms[1]
                    + ": '"
                    + ("', '").join(self.__register.keys())
                    + "'"
                )
            raise tools.UserException(cause)
        # Instantiation
        return self.__register[name](*args, **kwargs)


# ---------------------------------------------------------------------------- #
# Simple list of "<key>:<value>" into dictionary parser


def parse_keyval_auto_convert(val: str) -> object:
    """
    Infer and convert the type represented by a string.

    Conversion is attempted in this order: boolean literals, integer, float, then
    string.

    Parameters
    ----------
    val : str
        String value to convert.

    Returns
    -------
    object
        Converted value, or ``val`` unchanged if no non-string type matches.
    """
    # Try guess 'bool'
    low = val.lower()
    if low == "false":
        return False
    elif low == "true":
        return True
    # Try guess number
    for cls in (int, float):
        try:
            return cls(val)
        except ValueError:
            continue
    # Else guess string
    return val


def parse_keyval(
    list_keyval: list[str], defaults: dict[str, object] | None = None
) -> dict[str, object]:
    """
    Parse ``<key>:<value>`` strings into a typed dictionary.

    This helper is used for command-line options such as
    ``--gar-args lr:0.01``. Keys present in ``defaults`` are converted to the
    type of their default value; other keys are converted by
    :func:`parse_keyval_auto_convert`.

    Parameters
    ----------
    list_keyval : list[str]
        Entries formatted as ``<key>:<value>``.
    defaults : dict[str, object] | None, optional
        Default key/value mappings. These defaults are also used for type
        inference and are copied into the returned dictionary when the
        corresponding key is not explicitly provided.

    Returns
    -------
    dict[str, object]
        Parsed key/value pairs with converted values.

    Raises
    ------
    tools.UserException
        If an entry is malformed, a key is provided more than once, or
        conversion to a default value's type fails.

    Example
    -------

    >>> parse_keyval(["lr:0.01", "batch:32"], defaults={"lr": 0.1})
    {'lr': 0.01, 'batch': 32}
    >>> parse_keyval(["debug:true", "workers:4"], defaults={})
    {'debug': True, 'workers': 4}
    """
    if defaults is None:
        defaults = {}
    parsed = {}
    # Parsing
    sep = ":"
    for entry in list_keyval:
        pos = entry.find(sep)
        if pos < 0:
            raise tools.UserException(
                "Expected list of "
                + repr("<key>:<value>")
                + ", got "
                + repr(entry)
                + " as one entry"
            )
        key = entry[:pos]
        if key in parsed:
            raise tools.UserException(
                "Key "
                + repr(key)
                + " had already been specified with value "
                + repr(parsed[key])
            )
        val = entry[pos + len(sep) :]
        # Guess/assert type constructibility
        if key in defaults:
            try:
                cls = type(defaults[key])
                if cls is bool:  # Special case
                    val = val.lower() not in ("", "0", "n", "false")
                else:
                    val = cls(val)
            except Exception:
                raise tools.UserException(
                    "Required key "
                    + repr(key)
                    + " expected a value of type "
                    + repr(getattr(type(defaults[key]), "__name__", "<unknown>"))
                )
        else:
            val = parse_keyval_auto_convert(val)
        # Bind (converted) value to associated key
        parsed[key] = val
    # Add default values (done first to be able to force a given type with 'required')
    for key in defaults:
        if key not in parsed:
            parsed[key] = defaults[key]
    # Return final dictionary
    return parsed


# ---------------------------------------------------------------------------- #
# Basic "full-qualification" string builder for a given instance/class


def fullqual(obj: object) -> str:
    """
    Return a class or instance's fully qualified name.

    Parameters
    ----------
    obj : object
        Class or instance to describe.

    Returns
    -------
    str
        Fully qualified class name. Instances are prefixed with
        ``"instance of "``.

    Example
    -------

    >>> fullqual(str)
    'builtins.str'
    >>> fullqual(pathlib.Path("."))
    'instance of pathlib.PosixPath'
    """
    # Prelude
    if isinstance(obj, type):
        prelude = ""
    else:
        prelude = "instance of "
        obj = type(obj)
    # Rebuilding
    return "%s%s.%s" % (
        prelude,
        getattr(obj, "__module__", "<unknown module>"),
        getattr(obj, "__qualname__", "<unknown name>"),
    )


# ---------------------------------------------------------------------------- #
# Basic "full-qualification" string builder for a given instance/class


def onetime(name: str | None = None) -> tuple[callable, callable]:
    """
    Create or retrieve a thread-safe one-shot flag.

    Parameters
    ----------
    name : str | None, optional
        Optional global flag name. Reusing the same name returns the same
        getter/setter pair.

    Returns
    -------
    tuple[callable, callable]
        ``(getter, setter)`` pair. ``getter`` returns whether the flag has been
        set, and ``setter`` permanently sets it to ``True``.
    """
    global onetime_register
    # Check if name exists
    if name is not None and name in onetime_register:
        return onetime_register[name]
    # Private variables
    lock = threading.Lock()
    value = False

    # Management closures
    def getter(*args, **kwargs):
        """
        Return whether the one-shot flag has been set.

        Parameters
        ----------
        *args : object
            Ignored positional arguments.
        **kwargs : object
            Ignored keyword arguments.

        Returns
        -------
        bool
            ``True`` once the associated setter has been called, otherwise
            ``False``.
        """
        nonlocal lock
        nonlocal value
        with lock:
            return value

    def setter(*args, **kwargs):
        """
        Set the one-shot flag to ``True``.

        Parameters
        ----------
        *args : object
            Ignored positional arguments.
        **kwargs : object
            Ignored keyword arguments.
        """
        nonlocal lock
        nonlocal value
        with lock:
            value = True

    # Register if need be, then return the management closures
    res = (getter, setter)
    if name is not None:
        onetime_register[name] = res
    return res


# Register for the onetime variables
onetime_register = dict()

# ---------------------------------------------------------------------------- #
# Plain context augmented with simple execution time measurement


class TimedContext(tools.Context):
    """Context manager that logs the elapsed runtime of a block."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the timed context.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``tools.Context``.
        **kwargs : object
            Keyword arguments forwarded to ``tools.Context``.
        """
        super().__init__(*args, **kwargs)

    def __enter__(self):
        """
        Start timing and enter the parent context.

        Returns
        -------
        object
            Value returned by ``tools.Context.__enter__``.
        """
        self._chrono = time.time()
        return super().__enter__()

    def __exit__(self, *args, **kwargs) -> None:
        """
        Stop timing, log elapsed time, and exit the parent context.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``tools.Context.__exit__``.
        **kwargs : object
            Keyword arguments forwarded to ``tools.Context.__exit__``.
        """
        # Measure elapsed runtime (in ns)
        runtime = (time.time() - self._chrono) * 1000000000.0
        # Recover ideal unit
        for unit in ("ns", "µs", "ms"):
            if runtime < 1000.0:
                break
            runtime /= 1000.0
        else:
            unit = "s"
        # Format and print string
        tools.trace(f"Execution time: {runtime:.3g} {unit}")
        # Forward call
        super().__exit__(*args, **kwargs)


# ---------------------------------------------------------------------------- #
# Switch to interactive mode, executing user inputs


def interactive(
    glbs: dict[str, object] | None = None,
    lcls: dict[str, object] | None = None,
    prompt: str = ">>> ",
    cprmpt: str = "... ",
) -> None:
    """
    Run a small interactive Python prompt.

    Press ``Ctrl+D`` or send an equivalent EOF signal to leave the prompt.

    Parameters
    ----------
    glbs : dict[str, object] | None, optional
        Globals dictionary used when evaluating commands. If ``None``, the
        caller's globals are used when available.
    lcls : dict[str, object] | None, optional
        Locals dictionary used when evaluating commands. If ``None``, the
        caller's locals are used when available, otherwise ``glbs`` is used.
    prompt : str, optional
        Prompt displayed for a new command.
    cprmpt : str, optional
        Prompt displayed while continuing a multi-line command.
    """
    # Recover caller's globals and locals
    try:
        caller = sys._getframe().f_back
    except Exception:
        caller = None
        if glbs is None:
            tools.warning(
                "Unable to recover caller's frame, locals and globals",
                context="interactive",
            )
    if glbs is None:
        if caller is not None and hasattr(caller, "f_globals"):
            glbs = caller.f_globals
        else:
            glbs = dict()
    if lcls is None:
        if caller is not None and hasattr(caller, "f_locals"):
            lcls = caller.f_locals
        else:
            lcls = glbs
    # Command input and execution
    command = ""
    statement = False
    while True:
        print(prompt if len(command) == 0 else cprmpt, end="", flush=True)
        try:
            # Input new line
            try:
                line = input()
                print(
                    "\033[A"
                )  # Trick to "advertise" new line on stdout after new line on stdin
            except BaseException as err:
                if any(isinstance(err, cls) for cls in (EOFError, KeyboardInterrupt)):
                    print()  # Since no new line was printed by pressing ENTER
                return
            # Handle expression
            if not statement:
                try:
                    res = eval(line, glbs, lcls)
                    if res is not None:
                        print(res)
                except SyntaxError:  # Heuristic that we are dealing with a statement
                    statement = True
            # Handle single or multi-line statement(s)
            if statement:
                if len(command) == 0:  # Just went through trying an expression
                    command = line
                    try:
                        exec(command, glbs, lcls)
                    except (
                        SyntaxError
                    ):  # Heuristic that we are dealing with a multi-line statement
                        continue
                elif len(line) > 0:
                    command += os.linesep + line
                    continue
                else:  # Multi-line statement is complete
                    exec(command, glbs, lcls)
        except Exception:
            with tools.Context("uncaught", "error"):
                traceback.print_exc()
        command = ""
        statement = False


# ---------------------------------------------------------------------------- #
# List non-standard, currently loaded module names and metadata.


def get_loaded_dependencies() -> list[tuple[str, str | None, int]]:
    """
    List currently loaded non-built-in root modules.

    Returns
    -------
    list[tuple[str, str | None, int]]
        Tuples of ``(root_module_name, version, flavor)``. ``version`` is the
        module's ``__version__`` attribute when present, otherwise ``None``.
        ``flavor`` is one of ``IS_STANDARD``, ``IS_SITE``, or ``IS_LOCAL``.

    Raises
    ------
    RuntimeError
        If Python's site-packages locations cannot be discovered on the current
        platform.
    """
    # Get the site-packages directories, and make "flavor"-detection closure
    path_sites = tuple(
        pathlib.Path(path)
        for path in site.getsitepackages() + [site.getusersitepackages()]
    )

    def flavor_of(path):
        path = pathlib.Path(path)
        for path_site in path_sites:
            try:
                path.relative_to(path_site)
                return get_loaded_dependencies.IS_SITE
            except ValueError:
                pass
        for path_site in path_sites:
            try:
                path.relative_to(path_site.parent)
                return get_loaded_dependencies.IS_STANDARD
            except ValueError:
                pass
        return get_loaded_dependencies.IS_LOCAL

    # Iterate over the loaded modules
    res = list()
    for name, module in sys.modules.items():
        # Skip non-root modules
        if "." in name:
            continue
        # Get module path (and so skip built-in modules)
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        # Get module version (if any)
        version = getattr(module, "__version__", None)
        # Get module "flavor"
        flavor = flavor_of(path)
        # Store entry
        res.append((name, version, flavor))
    # Return found root modules
    return res


# Register constants
get_loaded_dependencies.IS_STANDARD = 0
get_loaded_dependencies.IS_SITE = 1
get_loaded_dependencies.IS_LOCAL = 2

# ---------------------------------------------------------------------------- #
# Find the x maximizing a function y = f(x), with (x, y) ∊ ℝ⁺× ℝ


def line_maximize(
    scape: callable,
    evals: int = 16,
    start: float = 0.0,
    delta: float = 1.0,
    ratio: float = 0.8,
) -> float:
    """
    Best-effort argmax search for a scalar function on non-negative inputs.

    The search first expands while values improve, then contracts the step size to
    refine the best point found within the evaluation budget.

    Parameters
    ----------
    scape : callable
        Function to maximize. It is called with non-negative ``float`` values and
        must return comparable scores.
    evals : int, optional
        Maximum number of function evaluations.
    start : float, optional
        Initial non-negative point to evaluate.
    delta : float, optional
        Initial positive step size.
    ratio : float, optional
        Step contraction ratio, expected to be between ``0.5`` and ``1.0``
        excluded.

    Returns
    -------
    float
        Best point found under the evaluation budget.
    """
    # Variable setup
    best_x = start
    best_y = scape(best_x)
    evals -= 1
    # Expansion phase
    while evals > 0:
        prop_x = best_x + delta
        prop_y = scape(prop_x)
        evals -= 1
        # Check if best
        if prop_y > best_y:
            best_y = prop_y
            best_x = prop_x
            delta *= 2
        else:
            delta *= ratio
            break
    # Contraction phase
    while evals > 0:
        if prop_x < best_x:
            prop_x += delta
        else:
            x = prop_x - delta
            while x < 0:
                x = (x + prop_x) / 2
            prop_x = x
        prop_y = scape(prop_x)
        evals -= 1
        # Check if best
        if prop_y > best_y:
            best_y = prop_y
            best_x = prop_x
        # Reduce delta
        delta *= ratio
    # Return found maximizer
    return best_x


# ---------------------------------------------------------------------------- #
# Simple generator on the pairs (x, y) of an indexable such that index x < index y


def pairwise(data: list | tuple):
    """
    Yield unordered pairs from an indexable collection.

    Parameters
    ----------
    data : list | tuple
        Indexable collection such as a ``list`` or ``tuple``.

    Yields
    ------
    tuple
        Tuples ``(data[i], data[j])`` for every ``i < j``.

    Example
    -------

    >>> list(pairwise([1, 2, 3]))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(pairwise("ab"))
    [('a', 'b')]
    """
    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (data[i], data[j])


# ---------------------------------------------------------------------------- #
# Simple duration helpers


def localtime() -> str:
    """
    Return the current local time formatted for logs.

    Returns
    -------
    str
        Local time as ``YYYY/MM/DD HH:MM:SS``.
    """
    lt = time.localtime()
    return f"{lt.tm_year:04}/{lt.tm_mon:02}/{lt.tm_mday:02} {lt.tm_hour:02}:{lt.tm_min:02}:{lt.tm_sec:02}"


def deltatime_point() -> int:
    """
    Capture an opaque point in monotonic time.

    Returns
    -------
    int
        Monotonic timestamp rounded to seconds. The value is intended for use
        with :func:`deltatime_format`.
    """
    point = time.monotonic_ns()
    return (point + 5 * 10**8) // 10**9


def deltatime_format(a: int, b: int) -> tuple[int, str]:
    """
    Compute and format elapsed time between two captured points.

    Parameters
    ----------
    a : int
        Earlier point returned by :func:`deltatime_point`.
    b : int
        Later point returned by :func:`deltatime_point`.

    Returns
    -------
    tuple[int, str]
        Tuple ``(seconds, text)`` containing elapsed seconds and a
        human-readable duration string.
    """
    # Elapsed time (in seconds)
    t = b - a
    # Elapsed time (formatted)
    d = t
    s = d % 60
    d //= 60
    m = d % 60
    d //= 60
    h = d % 24
    d //= 24
    # Return elapsed time
    return t, f"{d} day(s), {h} hour(s), {m} min(s), {s} sec(s)"
