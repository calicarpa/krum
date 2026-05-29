"""Tests for tools.pytorch utilities."""

import unittest

import torch

from krum.tools.pytorch import flatten, relink


class RelinkTest(unittest.TestCase):
    """Test relink."""

    def test_modifying_common_updates_tensors(self) -> None:
        """Modifying the common tensor updates the relinked tensors."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0, 5.0])
        common = torch.zeros(5)

        relink([t1, t2], common)
        common[0] = 99.0

        self.assertEqual(t1[0].item(), 99.0)

    def test_modifying_tensor_updates_common(self) -> None:
        """Modifying a relinked tensor updates the common tensor."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0, 5.0])
        common = torch.zeros(5)

        relink([t1, t2], common)
        t1[1] = 88.0

        self.assertEqual(common[1].item(), 88.0)

    def test_linked_tensors_attribute(self) -> None:
        """The common tensor has a linked_tensors attribute."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0])
        common = torch.zeros(3)

        result = relink([t1, t2], common)

        self.assertIs(result, common)
        self.assertEqual(len(result.linked_tensors), 2)

    def test_different_shapes(self) -> None:
        """Relink handles tensors of different shapes."""
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([5.0])
        common = torch.zeros(5)

        relink([t1, t2], common)
        common[0] = 10.0
        common[3] = 20.0

        self.assertEqual(t1[0, 0].item(), 10.0)
        self.assertEqual(t1[1, 1].item(), 20.0)

    def test_relink_overwrites_original_data(self) -> None:
        """Relink replaces tensor data with views into the common buffer."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0])
        common = torch.tensor([10.0, 20.0, 30.0])

        relink([t1, t2], common)

        expected = torch.tensor([10.0, 20.0, 30.0])
        self.assertTrue(torch.equal(common, expected))
        self.assertEqual(t1[0].item(), 10.0)
        self.assertEqual(t2[0].item(), 30.0)

    def test_relink_preserves_dtype(self) -> None:
        """Relink preserves the dtype of the common tensor."""
        t1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t2 = torch.tensor([3.0], dtype=torch.float64)
        common = torch.zeros(3, dtype=torch.float64)

        relink([t1, t2], common)

        self.assertEqual(t1.dtype, torch.float64)
        self.assertEqual(t2.dtype, torch.float64)


class FlattenTest(unittest.TestCase):
    """Test flatten."""

    def test_flatten_returns_shared_memory(self) -> None:
        """Flatten returns a tensor sharing memory with the originals."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0, 5.0])

        flat = flatten([t1, t2])
        flat[0] = 99.0

        self.assertEqual(t1[0].item(), 99.0)

    def test_flatten_concatenates_correctly(self) -> None:
        """Flatten concatenates tensor data in order."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])

        flat = flatten([t1, t2])

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(torch.equal(flat, expected))

    def test_flatten_preserves_linked_tensors(self) -> None:
        """Flatten result has linked_tensors attribute."""
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])

        flat = flatten([t1, t2])

        self.assertEqual(len(flat.linked_tensors), 2)

    def test_flatten_single_tensor(self) -> None:
        """Flatten works with a single tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])

        flat = flatten([t])
        flat[0] = 99.0

        self.assertEqual(t[0].item(), 99.0)

    def test_flatten_modifying_original_updates_flat(self) -> None:
        """Modifying an original tensor after flatten updates the flat view."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])

        flat = flatten([t1, t2])
        t1[0] = 77.0
        t2[1] = 88.0

        self.assertEqual(flat[0].item(), 77.0)
        self.assertEqual(flat[3].item(), 88.0)

    def test_flatten_preserves_dtype(self) -> None:
        """Flatten result has the same dtype as input tensors."""
        t1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t2 = torch.tensor([3.0], dtype=torch.float64)

        flat = flatten([t1, t2])

        self.assertEqual(flat.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
