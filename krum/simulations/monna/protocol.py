"""Round-level protocol functions for MoNNA."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch

from krum.primitives import Model
from krum.primitives.aggregators import NearestNeighbor
from krum.primitives.attacks import Attack

from .config import MonnaConfig
from .state import MonnaState

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MonnaRoundResult:
    """Result of one MoNNA round."""

    state: MonnaState
    honest_gradients: torch.Tensor
    byzantine_vectors: torch.Tensor
    mixed_vectors: torch.Tensor
    losses: torch.Tensor


def _copy_parameters(model: Model, parameters: torch.Tensor) -> None:
    """Copy a flat parameter vector into a model."""
    with torch.no_grad():
        model.parameters.copy_(parameters)


def compute_worker_gradients(
    model: Model, parameters: torch.Tensor, batches: Sequence[Batch], loss_fn: LossFn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one real PyTorch gradient per honest worker.

    The same ``Model`` instance is reused sequentially. Before each backward
    pass, the worker's flat parameter vector is copied into the model.

    Args:
        model: Shared model wrapper used to compute gradients.
        parameters: Worker parameter vectors of shape ``(h, d)``.
        batches: One ``(inputs, targets)`` batch per honest worker.
        loss_fn: Loss function applied to ``model.module(inputs)`` and targets.

    Returns:
        Pair ``(gradients, losses)`` with shapes ``(h, d)`` and ``(h,)``.
    """
    if len(batches) != parameters.shape[0]:
        raise ValueError(f"Expected {parameters.shape[0]} batches, got {len(batches)!r}")

    gradients: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for worker_parameters, (inputs, targets) in zip(parameters, batches, strict=True):
        _copy_parameters(model, worker_parameters)
        model.module.zero_grad(set_to_none=True)
        loss = loss_fn(model.module(inputs), targets)
        loss.backward()
        gradients.append(model.gradients.detach().clone())
        losses.append(loss.detach())

    return torch.stack(gradients), torch.stack(losses)


def compute_momentum(previous: torch.Tensor, gradients: torch.Tensor, *, beta: float) -> torch.Tensor:
    """Compute the MoNNA worker-side Polyak momentum vectors."""
    return previous.mul(beta).add(gradients, alpha=1.0 - beta)


def _coordination_candidates(
    honest_vectors: torch.Tensor,
    byzantine_vectors: torch.Tensor,
    *,
    worker_index: int,
    f: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build the ``n - f`` vectors available to one honest worker during coordination."""
    num_honest = honest_vectors.shape[0]
    num_nodes = num_honest + f
    num_received = num_nodes - f - 1
    if num_received < 0:
        raise ValueError(
            "Expected enough nodes to average n - 2f values, "
            f"got num_honest={num_honest!r} and f={f!r}"
        )

    all_vectors = torch.cat([honest_vectors, byzantine_vectors], dim=0)
    other_indices = torch.cat(
        [
            torch.arange(0, worker_index, device=honest_vectors.device),
            torch.arange(worker_index + 1, num_nodes, device=honest_vectors.device),
        ]
    )
    permutation = torch.randperm(other_indices.numel(), generator=generator, device=honest_vectors.device)
    received_indices = other_indices[permutation[:num_received]]
    return torch.cat([honest_vectors[worker_index].unsqueeze(0), all_vectors[received_indices]], dim=0)


def mix_each_worker(
    honest_vectors: torch.Tensor,
    byzantine_vectors: torch.Tensor,
    *,
    f: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run one MoNNA coordination mixing step for every honest worker.

    Each honest worker waits for ``n - f - 1`` received vectors selected from
    a random permutation of the other ``n - 1`` nodes, combines them with its
    own vector, and runs nearest-neighbor averaging over that local set of
    ``n - f`` vectors. The NNA rule then discards ``f`` vectors, so the
    returned average is computed over ``n - 2f`` vectors.

    Args:
        honest_vectors: Honest worker vectors of shape ``(n - f, d)``.
        byzantine_vectors: Byzantine vectors of shape ``(f, d)``.
        f: Number of Byzantine vectors to discard.
        generator: Optional PyTorch random generator for reproducible receive selection.

    Returns:
        Mixed vectors, one per honest worker, shape ``(n - f, d)``.
    """
    if honest_vectors.ndim != 2:
        raise ValueError(f"Expected honest vectors with shape (h, d), got {tuple(honest_vectors.shape)!r}")
    if byzantine_vectors.ndim != 2:
        raise ValueError(f"Expected Byzantine vectors with shape (b, d), got {tuple(byzantine_vectors.shape)!r}")
    if byzantine_vectors.shape[0] != f:
        raise ValueError(f"Expected {f!r} Byzantine vectors, got {byzantine_vectors.shape[0]!r}")
    if byzantine_vectors.shape[1:] != honest_vectors.shape[1:]:
        raise ValueError(
            f"Expected Byzantine vector shape (*, {honest_vectors.shape[1]}), got {tuple(byzantine_vectors.shape)!r}"
        )

    aggregator = NearestNeighbor(n=honest_vectors.shape[0], f=f)
    mixed = []
    for worker_index, pivot in enumerate(honest_vectors):
        candidates = _coordination_candidates(
            honest_vectors, byzantine_vectors, worker_index=worker_index, f=f, generator=generator
        )
        mixed.append(aggregator.aggregate(candidates, pivot=pivot))
    return torch.stack(mixed)


def run_round(
    state: MonnaState,
    *,
    config: MonnaConfig,
    model: Model,
    batches: Sequence[Batch],
    loss_fn: LossFn,
    attack: Attack | None = None,
) -> MonnaRoundResult:
    """Run one synchronous MoNNA round with real honest-worker training.

    Args:
        state: Current distributed state.
        config: MoNNA configuration.
        model: Shared model used sequentially for gradient computation.
        batches: One local batch per honest worker.
        loss_fn: Loss function.
        attack: Optional attack that observes honest momentum vectors and
            produces Byzantine vectors.

    Returns:
        Round result containing the next state and trace tensors.
    """
    if state.parameters.shape[0] != config.num_honest:
        raise ValueError(f"Expected {config.num_honest} honest states, got {state.parameters.shape[0]!r}")
    if config.num_byzantine and attack is None:
        raise ValueError("An attack is required when num_byzantine > 0")

    gradients, losses = compute_worker_gradients(model, state.parameters, batches, loss_fn)
    momentum = compute_momentum(state.momentum, gradients, beta=config.beta)

    if config.num_byzantine == 0:
        byzantine_vectors = momentum.new_empty((0, momentum.shape[1]))
    else:
        byzantine_vectors = attack(momentum, num_byzantine=config.num_byzantine)

    mixed = mix_each_worker(momentum, byzantine_vectors, f=config.num_byzantine)
    next_parameters = state.parameters - config.learning_rate * mixed
    next_state = MonnaState(
        parameters=next_parameters.detach().clone(), momentum=momentum.detach().clone(), step=state.step + 1
    )
    return MonnaRoundResult(
        state=next_state,
        honest_gradients=gradients,
        byzantine_vectors=byzantine_vectors.detach().clone(),
        mixed_vectors=mixed.detach().clone(),
        losses=losses,
    )


def next_batches(iterators: Sequence[Iterator[Batch]]) -> list[Batch]:
    """Pull one batch from each worker iterator."""
    return [next(iterator) for iterator in iterators]


def run_simulation(
    state: MonnaState,
    *,
    config: MonnaConfig,
    model: Model,
    data: Sequence[Iterable[Batch]],
    loss_fn: LossFn,
    rounds: int,
    attack: Attack | None = None,
) -> list[MonnaRoundResult]:
    """Run several MoNNA rounds over per-worker data streams.

    Args:
        state: Initial state.
        config: MoNNA configuration.
        model: Shared model used sequentially for gradient computation.
        data: One iterable of local batches per honest worker.
        loss_fn: Loss function.
        rounds: Number of rounds to execute.
        attack: Optional Byzantine attack.

    Returns:
        One result per executed round.
    """
    if rounds < 0:
        raise ValueError(f"Expected non-negative rounds, got {rounds!r}")
    if len(data) != config.num_honest:
        raise ValueError(f"Expected {config.num_honest} data streams, got {len(data)!r}")

    iterators = [iter(worker_data) for worker_data in data]
    results = []
    current = state
    for _ in range(rounds):
        result = run_round(
            current,
            config=config,
            model=model,
            batches=next_batches(iterators),
            loss_fn=loss_fn,
            attack=attack,
        )
        results.append(result)
        current = result.state
    return results
