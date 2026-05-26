"""PyTorch utility functions."""

import torch


def relink(tensors: list[torch.Tensor], common: torch.Tensor) -> torch.Tensor:
    """Relink tensors to share a common contiguous memory storage.

    After relinking, modifying ``common`` or any individual tensor reflects
    in all others — they share the same underlying buffer.

    Args:
        tensors: Tensors to relink. All must have the same dtype.
        common: Flat tensor of sufficient size to use as underlying storage.

    Returns:
        The common tensor, with a ``linked_tensors`` attribute pointing
        to the original tensors.
    """
    offset = 0
    for tensor in tensors:
        end = offset + tensor.numel()
        tensor.data = common[offset:end].view(tensor.shape)
        offset = end
    common.linked_tensors = tensors
    return common


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Flatten tensors into a single contiguous tensor sharing memory.

    Args:
        tensors: Tensors to flatten. All must have the same dtype.

    Returns:
        Flat tensor containing all data from input tensors.
        Modifications to the flat tensor reflect in the originals.
    """
    common = torch.cat([t.view(-1) for t in tensors])
    return relink(tensors, common)
