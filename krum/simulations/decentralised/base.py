"""Base class for decentralised Byzantine-resilient learning simulations."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypedDict

import torch

from krum.primitives import Model
from krum.primitives.aggregators import Aggregator
from krum.primitives.attacks import Attack

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class StepResult(TypedDict):
    """Snapshot returned by :meth:`DecentralisedSimulation.step` for one round."""

    step: int
    parameters: torch.Tensor
    momentum: torch.Tensor
    honest_gradients: torch.Tensor
    local_parameters: torch.Tensor
    byzantine_parameters: torch.Tensor
    mixed_parameters: torch.Tensor
    losses: torch.Tensor


class DecentralisedSimulation(ABC):
    """Base for decentralised momentum-SGD simulations with per-worker model mixing.

    Each honest worker holds its own flat parameter vector — one row of
    :attr:`parameters`. A round runs a local momentum-SGD step, then a mixing
    phase in which every worker replaces its model with an aggregate of the
    models it *received* this round.

    What varies between decentralised protocols is **which models a worker
    receives** — the communication topology. That is the single abstract seam:
    subclasses implement :meth:`gather_received_models` to build each worker's
    received set (e.g. MoNNA's reach modes, or a pull-based sampling rule),
    while this base owns the local update, the mixing loop, the Byzantine
    generation hook, state, snapshots, and the multi-round :meth:`run` driver.

    :meth:`run` may be called repeatedly to continue training: all state
    (:attr:`parameters`, :attr:`momentum`, :attr:`step_index`, and the worker
    data streams) lives on the instance and persists across calls. Callers that
    run more rounds than a finite stream provides should cycle their streams.
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
        aggregator: type[Aggregator],
        beta: float = 0.99,
        attack: type[Attack] | None = None,
        attack_kwargs: dict[str, Any] | None = None,
        aggregator_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the shared decentralised state.

        Args:
            model: Model wrapper whose flat parameters seed every worker.
            data: One batch stream per honest worker; ``len(data)`` must equal
                ``num_honest``.
            loss_fn: Callable mapping ``(predictions, targets)`` to a scalar loss.
            num_honest: Number of honest workers ``n - f``; must be at least 1.
            num_byzantine: Number of Byzantine workers ``f``; must be
                non-negative and smaller than ``num_honest``.
            learning_rate: Positive local step size.
            aggregator: :class:`~krum.primitives.aggregators.Aggregator` subclass
                whose ``aggregate`` classmethod mixes each worker's received
                models. Subclasses typically resolve a protocol default before
                passing it here.
            beta: Momentum coefficient in ``[0, 1)``. ``0`` reduces the local
                update to plain SGD.
            attack: :class:`~krum.primitives.attacks.Attack` subclass whose
                ``generate`` classmethod maps the honest models and ``f`` to the
                Byzantine models. Required when ``num_byzantine > 0``.
            attack_kwargs: Extra keyword arguments forwarded to
                ``attack.generate`` (e.g. ``{"std": 200.0}``). ``f`` is injected
                by the simulation.
            aggregator_kwargs: Extra keyword arguments forwarded to
                ``aggregator.aggregate`` (e.g. ``{"num_closest": ...}``). The
                ``pivot`` is injected per worker.
            seed: Optional integer seed for responder sampling.

        Raises:
            ValueError: If a worker count, learning rate, beta, or data length is
                out of range, or an attack is missing while ``num_byzantine > 0``.
            TypeError: If ``model`` or ``loss_fn`` has the wrong type, ``attack``
                is not an :class:`~krum.primitives.attacks.Attack` subclass,
                ``aggregator`` is not an
                :class:`~krum.primitives.aggregators.Aggregator` subclass, or
                ``seed`` is not an int or ``None``.
        """
        if num_honest < 1:
            raise ValueError(f"Expected at least one honest worker, got {num_honest!r}")
        if num_byzantine < 0:
            raise ValueError(f"Expected non-negative Byzantine worker count, got {num_byzantine!r}")
        if num_honest <= num_byzantine:
            raise ValueError(
                "Expected enough honest workers for decentralised model mixing, "
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
        if attack is not None and not (isinstance(attack, type) and issubclass(attack, Attack)):
            raise TypeError("Expected attack to be an Attack subclass")
        if not (isinstance(aggregator, type) and issubclass(aggregator, Aggregator)):
            raise TypeError("Expected aggregator to be an Aggregator subclass")
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
        self.attack_kwargs = attack_kwargs or {}
        self.aggregator = aggregator
        self.aggregator_kwargs = dict(aggregator_kwargs or {})
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
        self.parameters = model.parameters.detach().clone().repeat(num_honest, 1)
        self.momentum = torch.zeros_like(self.parameters)
        self.step_index = 0

    def step(self) -> StepResult:
        """Execute one decentralised training round.

        Runs one local momentum-SGD phase followed by one model-mixing phase
        over the per-worker received sets, then commits the resulting state.

        Returns:
            A snapshot dict of the round, as built by :meth:`commit_step`.
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
        """Execute several rounds and collect their snapshots.

        State persists on the instance, so successive calls continue training
        from where the previous call left off.

        Args:
            rounds: Number of rounds to run; must be non-negative.

        Returns:
            One snapshot per round, in execution order.

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
        """Generate the Byzantine model vectors injected into the mixing phase.

        Called once per round before the per-worker mixing loop. The same
        Byzantine models are then placed into received sets by
        :meth:`gather_received_models`. Protocols whose Byzantine replies depend
        on the recipient (e.g. recipient-specific attacks) override this to
        produce them inside ``gather_received_models`` instead.

        Args:
            local_parameters: Post-local-update honest models, one row per
                worker, passed to the attack.

        Returns:
            The Byzantine models, shape ``(f, d)``; empty when
            ``num_byzantine == 0``.
        """
        if self.num_byzantine == 0:
            return local_parameters.new_empty((0, local_parameters.shape[1]))
        assert self.attack is not None  # guaranteed by __init__ when num_byzantine > 0
        return self.attack.generate(local_parameters, f=self.num_byzantine, **self.attack_kwargs)

    def aggregate_over_received_nodes(
        self, local_parameters: torch.Tensor, byzantine_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Run the model-mixing phase over post-local-update parameter vectors.

        For each worker, builds its received set via the protocol-specific
        :meth:`gather_received_models`, then aggregates it.

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

    @abstractmethod
    def gather_received_models(
        self,
        honest_vectors: torch.Tensor,
        byzantine_parameters: torch.Tensor,
        *,
        worker_index: int,
    ) -> torch.Tensor:
        """Build the set of models received by one honest worker this round.

        This is the communication-topology seam: each decentralised protocol
        decides which models (honest and Byzantine) land in a worker's received
        set. Implementations should lead the set with the worker's own model so
        a pivot-anchored aggregator can rely on its position.

        Args:
            honest_vectors: Post-local-update honest models, one row per worker.
            byzantine_parameters: Byzantine models, shape ``(f, d)``, as produced
                by :meth:`generate_byzantine_models` (may be empty or ignored by
                protocols that generate Byzantine replies per recipient).
            worker_index: Index of the receiving honest worker.

        Returns:
            The received models for the worker, with its own model first.
        """

    def aggregate_received_models(self, candidates: torch.Tensor, *, pivot: torch.Tensor) -> torch.Tensor:
        """Aggregate the set of models one worker received.

        ``NearestNeighborAverage`` anchors on the worker's own model via ``pivot``;
        pivot-free aggregators (e.g. Krum, Median) absorb it through ``**specialized``.

        Args:
            candidates: The received models for one worker.
            pivot: The worker's own model, used as the distance reference.

        Returns:
            The single mixed model for the worker, shape ``(d,)``.
        """
        return self.aggregator.aggregate(candidates, pivot=pivot, **self.aggregator_kwargs)

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
