"""Class-based MoNNA simulation."""

from collections.abc import Callable, Iterable, Sequence

import torch

from krum.primitives import Model
from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StepResult = dict[str, int | torch.Tensor]


class MonnaSimulation:
    """MoNNA simulation runner.

    The object owns configuration, state, data streams, attack, and aggregator.
    Calling :meth:`step` executes one local training phase followed by one
    model-mixing phase over parameter vectors.
    """

    def __init__(
        self,
        *,
        model: Model,
        data: Sequence[Iterable[Batch]],
        loss_fn: LossFn,
        aggregator: Aggregator,
        num_honest: int,
        num_byzantine: int,
        learning_rate: float,
        beta: float = 0.99,
        attack: Attack | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a MoNNA simulation."""
        if num_honest < 1:
            raise ValueError(f"Expected at least one honest worker, got {num_honest!r}")
        if num_byzantine < 0:
            raise ValueError(f"Expected non-negative Byzantine worker count, got {num_byzantine!r}")
        if num_honest <= num_byzantine:
            raise ValueError(
                "Expected enough honest workers for MoNNA model mixing, "
                f"got num_honest={num_honest!r} and num_byzantine={num_byzantine!r}"
            )
        if learning_rate <= 0:
            raise ValueError(f"Expected positive learning rate, got {learning_rate!r}")
        if beta < 0 or beta >= 1:
            raise ValueError(f"Expected beta in [0, 1), got {beta!r}")
        if not isinstance(model, Model):
            raise TypeError(f"Expected model to be a Model, got {type(model).__name__}")
        if not callable(loss_fn):
            raise TypeError("Expected loss_fn to be callable")
        if len(data) != num_honest:
            raise ValueError(f"Expected {num_honest} data streams, got {len(data)!r}")
        if num_byzantine and attack is None:
            raise ValueError("An attack is required when num_byzantine > 0")
        if attack is not None and not callable(attack):
            raise TypeError("Expected attack to be callable")
        if not isinstance(aggregator, Aggregator):
            raise TypeError(f"Expected aggregator to be an Aggregator, got {type(aggregator).__name__}")
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"Expected seed to be an int or None, got {type(seed).__name__}")

        self.model = model
        self.worker_data_iterators = [iter(worker_data) for worker_data in data]
        self.loss_fn = loss_fn
        self.num_honest = num_honest
        self.num_byzantine = num_byzantine
        self.learning_rate = learning_rate
        self.beta = beta
        self.attack = attack
        self.aggregator = aggregator
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
        self.parameters = model.parameters.detach().clone().repeat(num_honest, 1)
        self.momentum = torch.zeros_like(self.parameters)
        self.step_index = 0

    def step(self) -> StepResult:
        """Execute one MoNNA training step and return a primitive snapshot dict."""
        batches = self.collect_worker_batches()
        gradients, losses = self.compute_honest_worker_gradients(batches)
        next_momentum = self.update_local_momentum(gradients)
        local_parameters = self.compute_local_parameter_updates(next_momentum)
        byzantine_parameters = self.generate_byzantine_models(local_parameters)
        mixed_parameters = self.aggregate_over_received_nodes(local_parameters, byzantine_parameters)
        return self.commit_step(
            momentum=next_momentum,
            parameters=mixed_parameters,
            honest_gradients=gradients,
            local_parameters=local_parameters,
            byzantine_parameters=byzantine_parameters,
            mixed_parameters=mixed_parameters,
            losses=losses,
        )

    def run(self, rounds: int) -> list[StepResult]:
        """Execute several MoNNA training steps."""
        if rounds < 0:
            raise ValueError(f"Expected non-negative rounds, got {rounds!r}")
        return [self.step() for _ in range(rounds)]

    def collect_worker_batches(self) -> list[Batch]:
        """Pull one local batch from every honest worker stream."""
        return [next(iterator) for iterator in self.worker_data_iterators]

    def compute_honest_worker_gradients(self, batches: Sequence[Batch]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients at each honest worker's current parameters."""
        gradients: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for worker_parameters, (inputs, targets) in zip(self.parameters, batches, strict=True):
            self.copy_parameters_to_model(worker_parameters)
            self.model.module.train()
            self.model.module.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model.module(inputs), targets)
            loss.backward()
            gradients.append(self.model.gradients.detach().clone())
            losses.append(loss.detach())

        return torch.stack(gradients), torch.stack(losses)

    def copy_parameters_to_model(self, parameters: torch.Tensor) -> None:
        """Copy one flat parameter vector into the shared model wrapper."""
        with torch.no_grad():
            self.model.parameters.copy_(parameters)

    def update_local_momentum(self, gradients: torch.Tensor) -> torch.Tensor:
        """Update each honest worker's local momentum vector."""
        return self.momentum.mul(self.beta).add(gradients, alpha=1.0 - self.beta)

    def compute_local_parameter_updates(self, momentum: torch.Tensor) -> torch.Tensor:
        """Compute ``theta_{t+1/2}`` before the model-mixing phase."""
        return self.parameters - self.learning_rate * momentum

    def generate_byzantine_models(self, local_parameters: torch.Tensor) -> torch.Tensor:
        """Generate Byzantine model vectors injected into the model-mixing phase."""
        if self.num_byzantine == 0:
            return local_parameters.new_empty((0, local_parameters.shape[1]))
        return self.attack(local_parameters, num_byzantine=self.num_byzantine)

    def aggregate_over_received_nodes(self, local_parameters: torch.Tensor, byzantine_parameters: torch.Tensor) -> torch.Tensor:
        """Run MoNNA model mixing over post-local-update parameter vectors."""
        mixed_parameters = []
        for worker_index, pivot in enumerate(local_parameters):
            candidates = self.gather_received_models(
                local_parameters,
                byzantine_parameters,
                worker_index=worker_index,
            )
            mixed_parameters.append(self.aggregate_received_models(candidates, pivot=pivot))
        return torch.stack(mixed_parameters)

    def gather_received_models(
        self,
        honest_vectors: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        *,
        worker_index: int,
    ) -> torch.Tensor:
        """Build the ``n - f`` set of models received by one honest worker."""
        all_vectors = torch.cat([honest_vectors, byzantine_parameters], dim=0)
        received_indices = self.select_received_model_indices(worker_index=worker_index, device=honest_vectors.device)
        return torch.cat([honest_vectors[worker_index].unsqueeze(0), all_vectors[received_indices]], dim=0)

    def select_received_model_indices(self, *, worker_index: int, device: torch.device) -> torch.Tensor:
        """Randomly select the ``n - f - 1`` nodes received by one honest worker."""
        num_nodes = self.num_honest + self.num_byzantine
        num_received = num_nodes - self.num_byzantine - 1
        other_indices = torch.cat([
            torch.arange(0, worker_index, device=device),
            torch.arange(worker_index + 1, num_nodes, device=device),
        ])
        permutation = torch.randperm(other_indices.numel(), generator=self.generator, device=device)
        return other_indices[permutation[:num_received]]

    def aggregate_received_models(self, candidates: torch.Tensor, *, pivot: torch.Tensor) -> torch.Tensor:
        """Aggregate the set of models one worker received.

        ``NearestNeighborAverage`` anchors on the worker's own model via ``pivot``;
        pivot-free aggregators (e.g. Krum, Median) ignore it.
        """
        try:
            return self.aggregator.aggregate(candidates, pivot=pivot)
        except TypeError:
            return self.aggregator.aggregate(candidates)

    def commit_step(
        self,
        *,
        momentum: torch.Tensor,
        parameters: torch.Tensor,
        honest_gradients: torch.Tensor,
        local_parameters: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        mixed_parameters: torch.Tensor,
        losses: torch.Tensor,
    ) -> StepResult:
        """Store the next simulation state and return a plain snapshot dict."""
        self.step_index += 1
        self.momentum = momentum.detach().clone()
        self.parameters = parameters.detach().clone()
        return {
            "step": self.step_index,
            "parameters": self.parameters.detach().clone(),
            "momentum": self.momentum.detach().clone(),
            "honest_gradients": honest_gradients.detach().clone(),
            "local_parameters": local_parameters.detach().clone(),
            "byzantine_parameters": byzantine_parameters.detach().clone(),
            "mixed_parameters": mixed_parameters.detach().clone(),
            "losses": losses.detach().clone(),
        }
