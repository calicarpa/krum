"""State containers for the MoNNA simulation."""

from dataclasses import dataclass

import torch

from krum.primitives import Model


@dataclass(frozen=True)
class MonnaState:
    """Distributed state for honest MoNNA workers.

    Args:
        parameters: Per-honest-worker parameter vectors of shape ``(h, d)``.
        momentum: Per-honest-worker momentum vectors of shape ``(h, d)``.
        step: Number of completed rounds.
    """

    parameters: torch.Tensor
    momentum: torch.Tensor
    step: int = 0

    def __post_init__(self) -> None:
        """Validate state tensor shapes."""
        if self.parameters.ndim != 2:
            raise ValueError(f"Expected parameters with shape (h, d), got {tuple(self.parameters.shape)!r}")
        if self.momentum.shape != self.parameters.shape:
            raise ValueError(
                f"Expected momentum shape {tuple(self.parameters.shape)!r}, got {tuple(self.momentum.shape)!r}"
            )
        if self.step < 0:
            raise ValueError(f"Expected non-negative step, got {self.step!r}")


def initial_state(model: Model, *, num_honest: int) -> MonnaState:
    """Create a MoNNA state where every honest worker starts from the model parameters.

    Args:
        model: Model providing the initial parameter vector.
        num_honest: Number of honest workers.

    Returns:
        Initial state with zero momentum.
    """
    if num_honest < 1:
        raise ValueError(f"Expected at least one honest worker, got {num_honest!r}")
    parameters = model.parameters.detach().clone().repeat(num_honest, 1)
    momentum = torch.zeros_like(parameters)
    return MonnaState(parameters=parameters, momentum=momentum)
