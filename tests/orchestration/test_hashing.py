"""Tests for the orchestration hashing primitives.

The invariants pinned here are the contract the orchestrator relies on to
decide what to re-run: a cosmetic edit must keep a job's key, and any change
reaching the job's behaviour through owned code must change it.
"""

import unittest
from textwrap import dedent
from typing import Any

from krum.orchestration.hashing import (
    Hash,
    Hasher,
    HashError,
    Location,
    Modules,
    callee_key,
    code_of,
    is_local_file,
    shallow_key,
    static_key,
)

OWNED = "krum_hashing_fixture"
OTHER = "third_party_fixture"
# Digest length in bytes, ample to name a job folder
DIGEST_SIZE = 16


def build(source: str, *, name: str = "target", module: str = OWNED, filename: str = "<fixture>") -> Any:
    """Compile a source snippet in a fresh namespace and return one of its objects.

    Args:
        source: The snippet to compile, dedented first.
        name: The name to pull out of the resulting namespace.
        module: The `__name__` the namespace reports, which drives ownership.
        filename: The compiled filename, which must never reach a hash.

    Returns:
        The named object, with a `__globals__` scoped to the snippet.
    """
    namespace: dict[str, Any] = {"__name__": module}
    exec(compile(dedent(source), filename, "exec"), namespace)  # noqa: S102
    return namespace[name]


def key(obj: Any, *, owned: tuple[str, ...] = (OWNED,)) -> Hash:
    """Hash a single object under the given owned module prefixes."""
    hasher = Hasher(owned)
    hasher.push(obj)
    return hasher.digest()


class LocationTest(unittest.TestCase):
    """Test the Location helper."""

    def test_fetch_round_trips_a_type(self) -> None:
        """A type's location fetches the type back."""
        self.assertIs(Location.of_type(Hasher).fetch(), Hasher)

    def test_str_renders_dotted_path(self) -> None:
        """A location renders as its dotted path."""
        self.assertEqual(str(Location("a.b", "C.d")), "a.b.C.d")

    def test_encoding_separates_module_from_qualname(self) -> None:
        """The encoded form can be parsed back, where a dotted one could not.

        `a.b.C.d` does not say whether the module is `a` or `a.b`; the colon
        does.
        """
        location = Location("a.b", "C.d")
        self.assertEqual(location.encode(), "a.b:C.d")
        decoded = Location.decode(location.encode())
        self.assertEqual(decoded.module, "a.b")
        self.assertEqual(decoded.qualname, "C.d")

    def test_encoding_round_trips_an_ambiguous_name(self) -> None:
        """Two locations that share a dotted form stay distinguishable."""
        first = Location("a", "b.C.d")
        second = Location("a.b", "C.d")
        self.assertEqual(str(first), str(second))
        self.assertNotEqual(first.encode(), second.encode())


class ShallowKeyTest(unittest.TestCase):
    """Test hashing one callable's own code, apart from what it refers to."""

    SOURCE = """
        def helper(x):
            return x * {factor}

        def target(n):
            return helper(n)
        """

    def test_body_change_changes_the_shallow_key(self) -> None:
        """A function's own edit changes its shallow key."""
        before = build("def target(n):\n    return n + 1\n")
        after = build("def target(n):\n    return n + 2\n")
        self.assertNotEqual(shallow_key(before), shallow_key(after))

    def test_a_comment_does_not_change_the_shallow_key(self) -> None:
        """Line provenance is excluded here too."""
        plain = build("def target(n):\n    return n + 1\n")
        shifted = build("# a note\n\ndef target(n):\n    return n + 1\n")
        self.assertEqual(shallow_key(plain), shallow_key(shifted))

    def test_a_changed_helper_does_not_change_the_caller(self) -> None:
        """A shallow key covers one function, which is what attributes a change.

        The full key folds in everything an experiment reaches, so it moves
        when a helper does; a shallow key stays put, so a change can be
        reported against the function it actually happened in.
        """
        before = build(self.SOURCE.format(factor=2))
        after = build(self.SOURCE.format(factor=3))
        self.assertNotEqual(key(before), key(after))
        self.assertEqual(shallow_key(before), shallow_key(after))

    def test_code_of_sees_through_wrappers(self) -> None:
        """Methods, class methods and wrapped functions all yield their code."""
        holder = build(
            """
            import functools

            class Holder:
                @classmethod
                def as_class_method(cls):
                    return 1

                @staticmethod
                def as_static_method():
                    return 2

                def as_method(self):
                    return 3

            def decorate(function):
                @functools.wraps(function)
                def wrapper(*args, **kwargs):
                    return function(*args, **kwargs)
                return wrapper

            @decorate
            def decorated():
                return 4
            """,
            name="Holder",
        )
        for accessor in ("as_class_method", "as_static_method", "as_method"):
            with self.subTest(accessor=accessor):
                self.assertIsNotNone(code_of(getattr(holder, accessor)))

    def test_code_of_returns_none_for_a_builtin(self) -> None:
        """A function implemented in C has no code object to find."""
        self.assertIsNone(code_of(len))

    def test_an_object_without_code_still_hashes(self) -> None:
        """Something with no code falls back on hashing the object."""
        self.assertEqual(len(shallow_key("not a callable")), DIGEST_SIZE)


class ModulesTest(unittest.TestCase):
    """Test module ownership by prefix, and by where a module was loaded from."""

    def test_owns_module_matches_prefix_and_submodules(self) -> None:
        """A prefix owns itself and its submodules, not a merely similar name."""
        modules = Modules(("krum",))
        self.assertTrue(modules.owns_module("krum"))
        self.assertTrue(modules.owns_module("krum.primitives.aggregators.krum"))
        self.assertFalse(modules.owns_module("krumble"))
        self.assertFalse(modules.owns_module("torch"))
        self.assertFalse(modules.owns_module(None))

    def test_local_test_owns_what_is_not_installed(self) -> None:
        """A local test owns this test module, loaded from the repository, and no installed one."""
        self.assertFalse(Modules(()).owns_module(__name__))
        local = Modules((), local=True)
        self.assertTrue(local.owns_module(__name__))
        self.assertFalse(local.owns_module("krum.orchestration.hashing"), "exclusions still win")
        self.assertFalse(local.owns_module("unittest"))
        self.assertFalse(local.owns_module("pytest"))
        self.assertFalse(local.owns_module("a_module_nobody_imported"))

    def test_local_file_is_judged_on_its_directory(self) -> None:
        """The standard library and site-packages are installed; the rest is local."""
        self.assertTrue(is_local_file(__file__))
        self.assertFalse(is_local_file(unittest.__file__))
        self.assertFalse(is_local_file(None))

    def test_tests_compare_by_what_they_own(self) -> None:
        """Equal tests are equal, and the local flag tells two apart."""
        self.assertEqual(Modules(("a",)), Modules(("a",)))
        self.assertNotEqual(Modules(("a",)), Modules(("a",), local=True))


class CalleeKeyTest(unittest.TestCase):
    """Test the hash a traced job records for each function it entered."""

    SOURCE = """
        SCALE = {scale}

        def helper(x):
            return x * {factor}

        def target(n):
            return helper(n) * SCALE
        """

    def test_a_changed_global_changes_the_callee_key(self) -> None:
        """A constant the function reads is part of its record, where its code alone is not."""
        before = build(self.SOURCE.format(scale=2, factor=1))
        after = build(self.SOURCE.format(scale=3, factor=1))
        self.assertEqual(shallow_key(before), shallow_key(after))
        self.assertNotEqual(callee_key(before), callee_key(after))

    def test_a_changed_helper_does_not_change_the_callee_key(self) -> None:
        """A function the code calls folds in as its name: it is recorded on its own."""
        before = build(self.SOURCE.format(scale=2, factor=1))
        after = build(self.SOURCE.format(scale=2, factor=5))
        self.assertEqual(callee_key(before), callee_key(after))

    def test_a_changed_default_changes_the_callee_key(self) -> None:
        """A default value lives on the function, not in its code, and is covered too."""
        before = build("def target(n, factor=2):\n    return n * factor\n")
        after = build("def target(n, factor=3):\n    return n * factor\n")
        self.assertNotEqual(callee_key(before), callee_key(after))

    def test_an_extension_module_is_hashed_by_its_file(self) -> None:
        """Native code has no bytecode: the file it was loaded from stands for it."""
        import _ctypes  # noqa: PLC0415

        self.assertEqual(callee_key(_ctypes), callee_key(_ctypes))
        self.assertEqual(len(callee_key(_ctypes)), DIGEST_SIZE)


class CosmeticChangeTest(unittest.TestCase):
    """Changes that must NOT change a key."""

    def test_comment_above_function_keeps_key(self) -> None:
        """Adding a comment above a function keeps its key.

        This is the invariant that keeps a micro-edit from re-running a sweep:
        `co_firstlineno` and the line table are excluded from the hash.
        """
        plain = build("""
            def target(n):
                return n + 1
            """)
        commented = build("""
            # a note for the reader
            # spanning two lines

            def target(n):
                return n + 1
            """)
        self.assertEqual(key(plain), key(commented))

    def test_inline_comment_and_blank_lines_keep_key(self) -> None:
        """Comments inside a body, and blank lines, keep the key."""
        plain = build("""
            def target(n):
                total = n + 1
                return total
            """)
        annotated = build("""
            def target(n):
                # running total

                total = n + 1

                return total  # done
            """)
        self.assertEqual(key(plain), key(annotated))

    def test_comment_above_class_keeps_key(self) -> None:
        """Adding a comment above a class keeps the key of its dependents.

        Regression guard for `__firstlineno__`, which Python 3.13 added to
        every class `__dict__`: left in, it would re-run every job depending on
        a class merely because a line was inserted above it.
        """
        plain = build("""
            class Aggregator:
                @classmethod
                def aggregate(cls, values):
                    return sum(values)

            def target(n):
                return Aggregator.aggregate(range(n))
            """)
        shifted = build("""
            # a note for the reader
            # spanning two lines

            class Aggregator:
                @classmethod
                def aggregate(cls, values):
                    return sum(values)

            def target(n):
                return Aggregator.aggregate(range(n))
            """)
        self.assertEqual(key(plain), key(shifted))

    def test_class_body_change_still_changes_key(self) -> None:
        """Skipping line provenance does not mask a real change to a class body."""
        source = """
            class Aggregator:
                @classmethod
                def aggregate(cls, values):
                    return sum(values) * {factor}

            def target(n):
                return Aggregator.aggregate(range(n))
            """
        before = build(source.format(factor=1))
        after = build(source.format(factor=2))
        self.assertNotEqual(key(before), key(after))

    def test_filename_keeps_key(self) -> None:
        """Moving a function to another file keeps its key."""
        source = """
            def target(n):
                return n + 1
            """
        here = build(source, filename="<here>")
        there = build(source, filename="<there>")
        self.assertEqual(key(here), key(there))

    def test_hashing_is_deterministic(self) -> None:
        """Hashing the same function twice yields the same digest."""
        target = build("""
            HELPER_SCALE = 3

            def helper(x):
                return x * HELPER_SCALE

            def target(n):
                return helper(n)
            """)
        self.assertEqual(key(target), key(target))


class CodeChangeTest(unittest.TestCase):
    """Changes to the callable itself that must change a key."""

    def test_body_change_changes_key(self) -> None:
        """Changing a function body changes its key."""
        before = build("""
            def target(n):
                return n + 1
            """)
        after = build("""
            def target(n):
                return n + 2
            """)
        self.assertNotEqual(key(before), key(after))

    def test_parameter_rename_changes_key(self) -> None:
        """Renaming a parameter changes the key, though the body is equivalent."""
        before = build("""
            def target(n):
                return 1
            """)
        after = build("""
            def target(n_workers):
                return 1
            """)
        self.assertNotEqual(key(before), key(after))

    def test_function_rename_changes_key(self) -> None:
        """Renaming the function changes the key."""
        before = build("""
            def target(n):
                return n
            """)
        after = build(
            """
            def renamed(n):
                return n
            """,
            name="renamed",
        )
        self.assertNotEqual(key(before), key(after))

    def test_default_value_change_changes_key(self) -> None:
        """Changing a default argument changes the key."""
        before = build("""
            def target(n=10):
                return n
            """)
        after = build("""
            def target(n=20):
                return n
            """)
        self.assertNotEqual(key(before), key(after))

    def test_docstring_change_changes_key(self) -> None:
        """Changing a docstring changes the key.

        Docstrings live in `co_consts`, and are kept rather than stripped: the
        conservative direction is a needless re-run, never a stale result.
        """
        before = build('''
            def target(n):
                """One thing."""
                return n
            ''')
        after = build('''
            def target(n):
                """Another thing."""
                return n
            ''')
        self.assertNotEqual(key(before), key(after))

    def test_nested_function_body_change_changes_key(self) -> None:
        """Changing the body of a nested function changes the key."""
        before = build("""
            def target(n):
                def inner(x):
                    return x + 1
                return inner(n)
            """)
        after = build("""
            def target(n):
                def inner(x):
                    return x + 2
                return inner(n)
            """)
        self.assertNotEqual(key(before), key(after))

    def test_closure_value_change_changes_key(self) -> None:
        """Changing a captured value changes the key."""
        make = build(
            """
            def make(scale):
                def target(n):
                    return n * scale
                return target
            """,
            name="make",
        )
        self.assertNotEqual(key(make(2)), key(make(3)))


class DependencyChangeTest(unittest.TestCase):
    """Changes reached transitively through owned code that must change a key."""

    HELPER_SOURCE = """
        def helper(x):
            return x * {factor}

        def target(n):
            return helper(n)
        """

    def test_helper_change_changes_key(self) -> None:
        """Changing a helper the callable names changes the callable's key.

        The helper is reached by resolving `co_names` against `__globals__`,
        with no need to execute the callable.
        """
        before = build(self.HELPER_SOURCE.format(factor=2))
        after = build(self.HELPER_SOURCE.format(factor=3))
        self.assertNotEqual(key(before), key(after))

    def test_constant_change_changes_key(self) -> None:
        """Changing a module-level constant the callable names changes its key."""
        source = """
            ROUNDS = {rounds}

            def target(n):
                return ROUNDS
            """
        before = build(source.format(rounds=100))
        after = build(source.format(rounds=200))
        self.assertNotEqual(key(before), key(after))

    def test_transitive_helper_change_changes_key(self) -> None:
        """A change two hops away, through another helper, changes the key."""
        source = """
            def deep(x):
                return x * {factor}

            def helper(x):
                return deep(x) + 1

            def target(n):
                return helper(n)
            """
        before = build(source.format(factor=2))
        after = build(source.format(factor=3))
        self.assertNotEqual(key(before), key(after))

    def test_helper_named_only_inside_a_lambda_changes_key(self) -> None:
        """A dependency named only by nested code is still reached.

        A lambda resolves its globals against the enclosing function's
        `__globals__`, so `_code_names` has to walk nested code objects.
        """
        source = """
            def helper(x):
                return x * {factor}

            def target(n):
                apply = lambda v: helper(v)
                return apply(n)
            """
        before = build(source.format(factor=2))
        after = build(source.format(factor=3))
        self.assertNotEqual(key(before), key(after))

    def test_method_change_changes_key(self) -> None:
        """Changing a method of a class the callable names changes the key."""
        source = """
            class Aggregator:
                @classmethod
                def aggregate(cls, values):
                    return sum(values) * {factor}

            def target(n):
                return Aggregator.aggregate(range(n))
            """
        before = build(source.format(factor=1))
        after = build(source.format(factor=2))
        self.assertNotEqual(key(before), key(after))

    def test_base_class_method_change_changes_key(self) -> None:
        """Changing an inherited method changes the key of the subclass."""
        source = """
            class Base:
                @classmethod
                def scale(cls):
                    return {factor}

            class Derived(Base):
                pass

            def target(n):
                return Derived.scale() * n
            """
        before = build(source.format(factor=1))
        after = build(source.format(factor=2))
        self.assertNotEqual(key(before), key(after))

    def test_slotted_class_hashes(self) -> None:
        """A class using __slots__ hashes without tripping on its descriptors."""
        target = build("""
            class Holder:
                __slots__ = ("value",)

                def __init__(self, value):
                    self.value = value

            def target(n):
                return Holder(n)
            """)
        self.assertEqual(len(key(target)), DIGEST_SIZE)


class OwnershipBoundaryTest(unittest.TestCase):
    """The recursion must stop outside the owned modules."""

    def test_unowned_helper_change_keeps_key(self) -> None:
        """A change inside an unowned module does not change the key.

        Dependency versions are deliberately not folded in here: they belong to
        the environment fingerprint, keyed on `uv.lock`.
        """
        source = """
            def helper(x):
                return x * {factor}

            def target(n):
                return helper(n)
            """
        before = build(source.format(factor=2), module=OTHER)
        after = build(source.format(factor=3), module=OTHER)
        self.assertEqual(key(before), key(after))

    def test_unowned_helper_change_changes_key_once_owned(self) -> None:
        """The same change does change the key once that module is owned."""
        source = """
            def helper(x):
                return x * {factor}

            def target(n):
                return helper(n)
            """
        before = build(source.format(factor=2), module=OTHER)
        after = build(source.format(factor=3), module=OTHER)
        self.assertNotEqual(key(before, owned=(OTHER,)), key(after, owned=(OTHER,)))

    def test_external_class_is_hashed_by_location(self) -> None:
        """An unowned class folds in as its location, not its body."""
        source = """
            class Thing:
                def method(self):
                    return {value}
            """
        before = build(source.format(value=1), name="Thing", module=OTHER)
        after = build(source.format(value=2), name="Thing", module=OTHER)
        self.assertEqual(key(before), key(after))
        # The same two classes differ once their module is owned
        self.assertNotEqual(key(before, owned=(OTHER,)), key(after, owned=(OTHER,)))


class CycleTest(unittest.TestCase):
    """Cyclic object graphs must terminate."""

    def test_mutual_recursion_terminates(self) -> None:
        """Two mutually recursive functions hash without recursing forever."""
        target = build("""
            def target(n):
                return 0 if n <= 0 else other(n - 1)

            def other(n):
                return target(n - 1)
            """)
        self.assertEqual(len(key(target)), DIGEST_SIZE)

    def test_self_recursion_terminates(self) -> None:
        """A self-recursive function hashes without recursing forever."""
        target = build("""
            def target(n):
                return 0 if n <= 0 else target(n - 1)
            """)
        self.assertEqual(len(key(target)), DIGEST_SIZE)


class ContainerTest(unittest.TestCase):
    """Container hashing must not depend on insertion order."""

    def test_dict_order_does_not_matter(self) -> None:
        """Two dicts with the same items hash alike regardless of order."""
        self.assertEqual(key({"a": 1, "b": 2}), key({"b": 2, "a": 1}))

    def test_set_order_does_not_matter(self) -> None:
        """Two sets with the same members hash alike."""
        self.assertEqual(key({1, 2, 3}), key({3, 1, 2}))

    def test_list_order_matters(self) -> None:
        """A list's order is part of its value."""
        self.assertNotEqual(key([1, 2]), key([2, 1]))


class StaticKeyTest(unittest.TestCase):
    """Test the run identity derived from a callable and its parameters."""

    SOURCE = """
        def target(n, f, seed=42):
            return n + f + seed
        """

    def setUp(self) -> None:
        """Build the fixture callable."""
        self.target = build(self.SOURCE)

    def test_same_inputs_give_same_key(self) -> None:
        """The same callable and parameters give the same key."""
        first = static_key(self.target, {"n": 10, "f": 2}, (OWNED,))
        second = static_key(self.target, {"n": 10, "f": 2}, (OWNED,))
        self.assertEqual(first, second)

    def test_parameter_value_change_changes_key(self) -> None:
        """Changing a hyper parameter value changes the key."""
        first = static_key(self.target, {"n": 10, "f": 2}, (OWNED,))
        second = static_key(self.target, {"n": 10, "f": 3}, (OWNED,))
        self.assertNotEqual(first, second)

    def test_defaults_are_applied(self) -> None:
        """Passing a default explicitly gives the same key as omitting it."""
        implicit = static_key(self.target, {"n": 10, "f": 2}, (OWNED,))
        explicit = static_key(self.target, {"n": 10, "f": 2, "seed": 42}, (OWNED,))
        self.assertEqual(implicit, explicit)

    def test_parameter_rename_changes_key(self) -> None:
        """Renaming a hyper parameter changes the key, at equal values."""
        renamed = build("""
            def target(n, n_byzantine, seed=42):
                return n + n_byzantine + seed
            """)
        first = static_key(self.target, {"n": 10, "f": 2}, (OWNED,))
        second = static_key(renamed, {"n": 10, "n_byzantine": 2}, (OWNED,))
        self.assertNotEqual(first, second)

    def test_class_parameter_is_keyed_by_location(self) -> None:
        """Two distinct classes passed as parameters give distinct keys."""
        source = """
            class Krum:
                pass

            class Average:
                pass
            """
        krum = build(source, name="Krum")
        average = build(source, name="Average")
        first = static_key(self.target, {"n": 10, "f": krum}, (OWNED,))
        second = static_key(self.target, {"n": 10, "f": average}, (OWNED,))
        self.assertNotEqual(first, second)

    def test_unknown_parameter_is_rejected(self) -> None:
        """A parameter the callable does not accept raises early."""
        with self.assertRaises(TypeError):
            static_key(self.target, {"n": 10, "f": 2, "nope": 1}, (OWNED,))

    def test_unhashable_dependency_is_reported(self) -> None:
        """A dependency that cannot be hashed reproducibly raises HashError."""
        target = build("""
            HANDLE = (value for value in (1, 2))

            def target(n):
                return HANDLE
            """)
        with self.assertRaises(HashError):
            static_key(target, {"n": 1}, (OWNED,))


if __name__ == "__main__":
    unittest.main()
