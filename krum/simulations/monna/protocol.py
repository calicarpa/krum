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


def mix_each_worker(honest_vectors: torch.Tensor, byzantine_vectors: torch.Tensor, *, f: int) -> torch.Tensor:
    """Run nearest-neighbor averaging independently for every honest worker.

    Args:
        honest_vectors: Honest worker vectors of shape ``(h, d)``.
        byzantine_vectors: Byzantine vectors of shape ``(b, d)``.
        f: Number of Byzantine vectors to discard.

    Returns:
        Mixed vectors, one per honest worker, shape ``(h, d)``.
    """
    if honest_vectors.ndim != 2:
        raise ValueError(f"Expected honest vectors with shape (h, d), got {tuple(honest_vectors.shape)!r}")
    if byzantine_vectors.ndim != 2:
        raise ValueError(f"Expected Byzantine vectors with shape (b, d), got {tuple(byzantine_vectors.shape)!r}")
    if byzantine_vectors.shape[1:] != honest_vectors.shape[1:]:
        raise ValueError(
            f"Expected Byzantine vector shape (*, {honest_vectors.shape[1]}), got {tuple(byzantine_vectors.shape)!r}"
        )

    candidates = torch.cat([honest_vectors, byzantine_vectors], dim=0)
    aggregator = NearestNeighbor(n=candidates.shape[0], f=f)
    mixed = []
    for pivot in honest_vectors:
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
