"""Tests for the FrozenDict immutable mapping."""

import builtins
import unittest

from krum.orchestration._frozendict import FrozenDict

# The internal-state probes below apply only to the pure-Python backport; on
# Python 3.15+ FrozenDict aliases the builtin, which exposes no such internals.
_USES_BUILTIN = FrozenDict is getattr(builtins, "frozendict", None)


class FrozenDictTest(unittest.TestCase):
    """Test FrozenDict mapping access, ordering, equality and hashing."""

    def test_mapping_access(self) -> None:
        """Items are accessible and the mapping reports its length."""
        frozen = FrozenDict(n=1, f=2)
        self.assertEqual(frozen["n"], 1)
        self.assertEqual(len(frozen), 2)
        self.assertIn("f", frozen)
        self.assertEqual(dict(frozen), {"n": 1, "f": 2})

    def test_iteration_preserves_insertion_order(self) -> None:
        """Iteration yields keys in insertion order, not sorted."""
        self.assertEqual(list(FrozenDict(b=1, a=2, c=3)), ["b", "a", "c"])

    def test_equality_is_order_independent(self) -> None:
        """Two mappings with the same items are equal regardless of order."""
        self.assertEqual(FrozenDict(n=1, f=2), FrozenDict(f=2, n=1))

    def test_equals_plain_dict(self) -> None:
        """A FrozenDict equals a plain dict with the same items."""
        self.assertEqual(FrozenDict(n=1, f=2), {"n": 1, "f": 2})

    def test_hash_is_order_independent(self) -> None:
        """Equal mappings hash alike regardless of insertion order."""
        self.assertEqual(hash(FrozenDict(n=1, f=2)), hash(FrozenDict(f=2, n=1)))

    def test_usable_as_dict_key(self) -> None:
        """Equal mappings address the same dict entry, regardless of order."""
        store = {FrozenDict(n=1, f=2): "x"}
        self.assertEqual(store[FrozenDict(f=2, n=1)], "x")

    def test_distinct_contents_are_distinct_keys(self) -> None:
        """Different contents are different keys."""
        store = {FrozenDict(n=1): "a", FrozenDict(n=2): "b"}
        self.assertEqual(len(store), 2)

    def test_construct_from_mapping_and_kwargs(self) -> None:
        """A FrozenDict can be built from a mapping plus keyword items."""
        self.assertEqual(dict(FrozenDict({"n": 1}, f=2)), {"n": 1, "f": 2})

    def test_construct_from_iterable_with_non_string_keys(self) -> None:
        """Construction accepts iterable pairs and arbitrary hashable keys."""
        frozen = FrozenDict([(1, "one"), ((2, 3), "tuple")])
        self.assertEqual(frozen[1], "one")
        self.assertEqual(frozen[(2, 3)], "tuple")

    def test_is_immutable(self) -> None:
        """Item and attribute assignment cannot mutate the mapping."""
        frozen = FrozenDict(n=1)
        with self.assertRaises(TypeError):
            frozen["n"] = 2  # ty: ignore[invalid-assignment]
        with self.assertRaises(AttributeError):
            frozen.extra = "value"  # type: ignore[attr-defined]
        self.assertFalse(hasattr(frozen, "_data"))
        if not _USES_BUILTIN:
            with self.assertRaises(AttributeError):
                frozen._FrozenDict__data = {"n": 2}  # type: ignore[attr-defined]

    def test_hash_remains_consistent_with_equality(self) -> None:
        """The cached hash stays consistent with equality."""
        frozen = FrozenDict(n=1)
        original_hash = hash(frozen)
        if not _USES_BUILTIN:
            # The backport caches its hash behind a read-only proxy, so the
            # underlying store cannot be mutated after the fact.
            with self.assertRaises(TypeError):
                frozen._FrozenDict__data["n"] = 2  # ty: ignore[unresolved-attribute]
        self.assertEqual(frozen, FrozenDict(n=1))
        self.assertEqual(hash(frozen), original_hash)
        self.assertEqual(hash(frozen), hash(FrozenDict(n=1)))

    def test_unhashable_value_raises_on_hash_not_construction(self) -> None:
        """Values may be unhashable; the mapping only raises when hashed."""
        frozen = FrozenDict(bad=[1, 2])  # construction is fine
        with self.assertRaises(TypeError):
            hash(frozen)

    def test_repr_shows_items(self) -> None:
        """The repr shows the contained items."""
        self.assertIn("'n': 1", repr(FrozenDict(n=1)))


if __name__ == "__main__":
    unittest.main()
