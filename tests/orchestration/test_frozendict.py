"""Tests for the FrozenDict immutable mapping."""

import copy
import pickle
import unittest

from krum.orchestration._frozendict import FrozenDict


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
            frozen["n"] = 2  # type: ignore[index]
        with self.assertRaises(AttributeError):
            frozen._FrozenDict__data = {"n": 2}  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            frozen.extra = "value"  # type: ignore[attr-defined]
        self.assertFalse(hasattr(frozen, "_data"))

    def test_hash_remains_consistent_with_equality(self) -> None:
        """The stored mapping cannot be changed after its hash is cached."""
        frozen = FrozenDict(n=1)
        original_hash = hash(frozen)
        with self.assertRaises(TypeError):
            frozen._FrozenDict__data["n"] = 2  # type: ignore[attr-defined,index]
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

    def test_union_returns_frozendict_and_uses_right_values(self) -> None:
        """Mapping unions preserve order and take values from the right."""
        left = FrozenDict(x=1, y=2)
        merged = left | {"y": 5, "z": 3}
        self.assertIsInstance(merged, FrozenDict)
        self.assertEqual(list(merged.items()), [("x", 1), ("y", 5), ("z", 3)])
        self.assertEqual({"w": 0, "x": 9} | left, FrozenDict(w=0, x=1, y=2))

    def test_in_place_union_rebinds_without_mutating_original(self) -> None:
        """The update operator creates a new mapping."""
        original = FrozenDict(x=1)
        merged = original
        merged |= {"y": 2}
        self.assertEqual(original, FrozenDict(x=1))
        self.assertEqual(merged, FrozenDict(x=1, y=2))
        self.assertIsNot(merged, original)

    def test_copy_returns_same_instance(self) -> None:
        """Shallow copies return the same immutable object."""
        frozen = FrozenDict(x=1)
        self.assertIs(frozen.copy(), frozen)
        self.assertIs(copy.copy(frozen), frozen)

    def test_deepcopy_copies_mutable_values(self) -> None:
        """Deep copies do not share nested mutable values."""
        frozen = FrozenDict(values=[1])
        cloned = copy.deepcopy(frozen)
        cloned["values"].append(2)
        self.assertEqual(frozen["values"], [1])
        self.assertEqual(cloned["values"], [1, 2])

    def test_pickle_round_trip(self) -> None:
        """Frozen mappings support pickling."""
        frozen = FrozenDict([(1, "one")], two=[2])
        restored = pickle.loads(pickle.dumps(frozen))
        self.assertEqual(restored, frozen)
        self.assertIsInstance(restored, FrozenDict)


if __name__ == "__main__":
    unittest.main()
