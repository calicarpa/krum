"""Class-based MoNNA simulation."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Literal

import torch

from krum.primitives import Model
from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StepResult = dict[str, int | torch.Tensor]
ByzantineReach = Literal["all", "sampled"]
# Attacks and aggregators are stateless callables: an attack maps the honest
# models and ``f`` to the Byzantine models; an aggregator maps the received
# models and a pivot to one mixed model.
AttackFn = Callable[..., torch.Tensor]
AggregatorFn = Callable[..., torch.Tensor]


class MonnaSimulation:
    """MoNNA simulation runner.

    The object owns configuration, state, data streams, attack, and aggregator.
    Calling :meth:`step` executes one local training phase followed by one
    model-mixing phase over parameter vectors.

    The model-mixing phase defaults to nearest-neighbor averaging over the
    ``n - 2f`` models closest to each worker's own model. Passing ``aggregator``
    overrides that rule (e.g. to compare other robust aggregators); MoNNA still
    owns the default and its sizing.

    Each honest worker receives ``n - f`` models per round: its own plus a set of
    responders. ``byzantine_reach`` controls how the Byzantine models are placed
    in that set:

    * ``"all"`` (default) injects every Byzantine model into every honest
      worker's received set and randomizes only the honest responders. This is
      the worst-case adversary used by the reference MoNNA implementation: the
      attack always reaches every worker, so the robustness it measures is not
      inflated by an adversary that randomly misses some workers.
    * ``"sampled"`` draws the responders uniformly from all other nodes, so a
      worker may receive anywhere from ``0`` to ``f`` Byzantine models. This
      models gossip where Byzantine reach is itself random.

    Both modes keep the received-set size at ``n - f``; only the Byzantine
    composition differs.
    """

    def __init__(
        self,
        *,
        model: Model,
        data: Sequence[Iterable[Batch]],
        loss_fn: LossFn,
        num_honest: int,
        num_byzantine: int,
        learning_rate: float,
        beta: float = 0.99,
        attack: AttackFn | None = None,
        aggregator: AggregatorFn | None = None,
        byzantine_reach: ByzantineReach = "all",
        seed: int | None = None,
    ) -> None:
        """Initialize a MoNNA simulation.

        Args:
            model: Model wrapper whose flat parameters seed every worker.
            data: One batch stream per honest worker; ``len(data)`` must equal
                ``num_honest``.
            loss_fn: Callable mapping ``(predictions, targets)`` to a scalar loss.
            num_honest: Number of honest workers ``n - f``; must be at least 1.
            num_byzantine: Number of Byzantine workers ``f``; must be
                non-negative and smaller than ``num_honest``.
            learning_rate: Positive local step size.
            beta: Momentum coefficient in ``[0, 1)``.
            attack: Callable mapping the honest models and ``f`` to the Byzantine
                models. Required when ``num_byzantine > 0``.
            aggregator: Optional model-mixing rule overriding the default
                nearest-neighbor average over the ``n - 2f`` closest models.
            byzantine_reach: ``"all"`` to inject every Byzantine model into every
                worker's received set, or ``"sampled"`` to draw responders
                uniformly from all other nodes.
            seed: Optional integer seed for responder sampling.

        Raises:
            ValueError: If a worker count, learning rate, beta, data length, or
                ``byzantine_reach`` is out of range, or an attack is missing
                while ``num_byzantine > 0``.
            TypeError: If ``model``, ``loss_fn``, ``attack``, ``aggregator``, or
                ``seed`` has the wrong type.
        """
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
        if aggregator is not None and not callable(aggregator):
            raise TypeError("Expected aggregator to be callable")
        if byzantine_reach not in ("all", "sampled"):
            raise ValueError(f"Expected byzantine_reach to be 'all' or 'sampled', got {byzantine_reach!r}")
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
        self.byzantine_reach = byzantine_reach
        # Each worker mixes over its n - f received models and keeps the
        # n - 2f closest to its own; num_honest - num_byzantine == n - 2f.
        self.aggregator = aggregator or partial(
            NearestNeighborAverage.aggregate, num_closest=num_honest - num_byzantine
        )
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
        self.parameters = model.parameters.detach().clone().repeat(num_honest, 1)
        self.momentum = torch.zeros_like(self.parameters)
        self.step_index = 0

    def step(self) -> StepResult:
        """Execute one MoNNA training step.

        Runs one local training phase followed by one model-mixing phase, then
        commits the resulting state.

        Returns:
            A snapshot dict of the step, as built by :meth:`commit_step`.
        """
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
        """Execute several MoNNA training steps.

        Args:
            rounds: Number of steps to run; must be non-negative.

        Returns:
            One snapshot dict per step, in execution order.

        Raises:
            ValueError: If ``rounds`` is negative.
        """
        if rounds < 0:
            raise ValueError(f"Expected non-negative rounds, got {rounds!r}")
        return [self.step() for _ in range(rounds)]

    def collect_worker_batches(self) -> list[Batch]:
        """Pull one local batch from every honest worker stream.

        Returns:
            One batch per honest worker, in worker order.
        """
        return [next(iterator) for iterator in self.worker_data_iterators]

    def compute_honest_worker_gradients(self, batches: Sequence[Batch]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients at each honest worker's current parameters.

        Args:
            batches: One batch per honest worker, in worker order.

        Returns:
            A tuple ``(gradients, losses)`` of stacked tensors, each with one row
            per honest worker.
        """
        gradients: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for worker_parameters, (inputs, targets) in zip(self.parameters, batches, strict=True):
            self.copy_parameters_to_model(worker_parameters)
            self.model.module.train()
            self.model.module.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model.module(inputs), targets)
            loss.backward()
            # ``zero_grad(set_to_none=True)`` drops each ``.grad`` and
            # ``backward`` allocates fresh ones, so the cached flat view must be
            # re-synchronised before reading ``model.gradients``.
            self.model.relink_gradients()
            gradients.append(self.model.gradients.detach().clone())
            losses.append(loss.detach())

        return torch.stack(gradients), torch.stack(losses)

    def copy_parameters_to_model(self, parameters: torch.Tensor) -> None:
        """Copy one flat parameter vector into the shared model wrapper.

        Args:
            parameters: Flat parameter vector of shape ``(d,)`` to load.
        """
        with torch.no_grad():
            self.model.parameters.copy_(parameters)

    def update_local_momentum(self, gradients: torch.Tensor) -> torch.Tensor:
        """Update each honest worker's local momentum vector.

        Args:
            gradients: Stacked honest gradients, one row per worker.

        Returns:
            The next momentum, with one row per honest worker.
        """
        return self.momentum.mul(self.beta).add(gradients, alpha=1.0 - self.beta)

    def compute_local_parameter_updates(self, momentum: torch.Tensor) -> torch.Tensor:
        """Compute ``theta_{t+1/2}`` before the model-mixing phase.

        Args:
            momentum: The next momentum, one row per honest worker.

        Returns:
            The post-local-update parameters, one row per honest worker.
        """
        return self.parameters - self.learning_rate * momentum

    def generate_byzantine_models(self, local_parameters: torch.Tensor) -> torch.Tensor:
        """Generate Byzantine model vectors injected into the model-mixing phase.

        Args:
            local_parameters: Post-local-update honest models, one row per
                worker, passed to the attack.

        Returns:
            The Byzantine models, shape ``(f, d)``; empty when
            ``num_byzantine == 0``.
        """
        if self.num_byzantine == 0:
            return local_parameters.new_empty((0, local_parameters.shape[1]))
        return self.attack(local_parameters, f=self.num_byzantine)

    def aggregate_over_received_nodes(
        self, local_parameters: torch.Tensor, byzantine_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Run MoNNA model mixing over post-local-update parameter vectors.

        Args:
            local_parameters: Post-local-update honest models, one row per
                worker.
            byzantine_parameters: Byzantine models, shape ``(f, d)``.

        Returns:
            The mixed models, one row per honest worker.
        """
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
        """Build the ``n - f`` set of models received by one honest worker.

        The worker's own model leads the set so a pivot-anchored aggregator can
        rely on its position; the remaining ``n - f - 1`` models are placed
        according to :attr:`byzantine_reach`.

        Args:
            honest_vectors: Post-local-update honest models, one row per worker.
            byzantine_parameters: Byzantine models, shape ``(f, d)``.
            worker_index: Index of the receiving honest worker.

        Returns:
            The ``n - f`` received models, with the worker's own model first.
        """
        own = honest_vectors[worker_index].unsqueeze(0)
        if self.byzantine_reach == "all":
            # Worst case: every Byzantine model reaches this worker; only the
            # honest responders are random. self (1) + n-2f-1 honest + f byz = n-f.
            responders = self.select_honest_responder_indices(worker_index=worker_index, device=honest_vectors.device)
            return torch.cat([own, honest_vectors[responders], byzantine_parameters], dim=0)
        # "sampled": responders drawn uniformly from all other nodes, so a worker
        # receives 0..f Byzantine models. self (1) + n-f-1 sampled = n-f.
        all_vectors = torch.cat([honest_vectors, byzantine_parameters], dim=0)
        received_indices = self.select_received_model_indices(worker_index=worker_index, device=honest_vectors.device)
        return torch.cat([own, all_vectors[received_indices]], dim=0)

    def select_honest_responder_indices(self, *, worker_index: int, device: torch.device) -> torch.Tensor:
        """Randomly select the ``n - 2f - 1`` other honest workers that respond to one worker.

        Used by the ``"all"`` reach mode, where the ``f`` Byzantine models are
        always included, so the honest responders fill the remaining slots.

        Args:
            worker_index: Index of the receiving honest worker, excluded from
                the selection.
            device: Device on which to build the index tensors.

        Returns:
            The selected honest responder indices, shape ``(n - 2f - 1,)``.
        """
        num_responders = self.num_honest - self.num_byzantine - 1
        other_indices = torch.cat([
            torch.arange(0, worker_index, device=device),
            torch.arange(worker_index + 1, self.num_honest, device=device),
        ])
        permutation = torch.randperm(other_indices.numel(), generator=self.generator, device=device)
        return other_indices[permutation[:num_responders]]

    def select_received_model_indices(self, *, worker_index: int, device: torch.device) -> torch.Tensor:
        """Randomly select the ``n - f - 1`` nodes received by one honest worker.

        Used by the ``"sampled"`` reach mode, where responders are drawn
        uniformly from every other node, honest or Byzantine.

        Args:
            worker_index: Index of the receiving honest worker, excluded from
                the selection.
            device: Device on which to build the index tensors.

        Returns:
            The selected node indices, shape ``(n - f - 1,)``.
        """
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
        pivot-free aggregators (e.g. Krum, Median) absorb it through ``**specialized``.

        Args:
            candidates: The ``n - f`` received models for one worker.
            pivot: The worker's own model, used as the distance reference.

        Returns:
            The single mixed model for the worker, shape ``(d,)``.
        """
        return self.aggregator(candidates, pivot=pivot)

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
        """Store the next simulation state and return a plain snapshot dict.

        Args:
            momentum: The next momentum to persist as state.
            parameters: The mixed models to persist as the next parameters.
            honest_gradients: Stacked honest gradients for the snapshot.
            local_parameters: Post-local-update honest models for the snapshot.
            byzantine_parameters: Byzantine models for the snapshot.
            mixed_parameters: Mixed models for the snapshot.
            losses: Per-worker losses for the snapshot.

        Returns:
            A snapshot dict with the step index and a detached clone of each
            tensor produced this step.
        """
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
