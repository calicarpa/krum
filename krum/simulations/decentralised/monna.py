"""Class-based MoNNA decentralised simulation."""

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import torch

from krum.primitives import Model
from krum.primitives.aggregators import Aggregator
from krum.primitives.aggregators.nearest_neighbor_average import NearestNeighborAverage
from krum.primitives.attacks import Attack
from krum.simulations.decentralised.base import Batch, DecentralisedSimulation, LossFn

ByzantineReach = Literal["all", "sampled"]


class MonnaSimulation(DecentralisedSimulation):
    """MoNNA simulation runner.

    Each round, every honest worker runs one local momentum-SGD step and then
    replaces its model with a nearest-neighbor average over the ``n - 2f``
    models closest to its own, drawn from the ``n - f`` models it received that
    round (its own plus a set of responders).

    ``byzantine_reach`` selects the adversary model used when forming those
    received sets in :meth:`gather_received_models`:

    * ``"all"`` is the worst case — every Byzantine model reaches every worker,
      and only the honest responders are randomized; the robustness measured is
      not inflated by an adversary that randomly misses some workers.
    * ``"sampled"`` draws responders uniformly from all other nodes, so a worker
      may receive anywhere from ``0`` to ``f`` Byzantine models, modelling
      gossip where Byzantine reach is itself random.

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
        attack: type[Attack] | None = None,
        attack_kwargs: dict[str, Any] | None = None,
        aggregator: type[Aggregator] | None = None,
        aggregator_kwargs: dict[str, Any] | None = None,
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
            attack: :class:`~krum.primitives.attacks.Attack` subclass whose
                ``generate`` classmethod maps the honest models and ``f`` to the
                Byzantine models. Required when ``num_byzantine > 0``.
            attack_kwargs: Extra keyword arguments forwarded to
                ``attack.generate`` (e.g. ``{"std": 200.0}``). ``f`` is injected
                by the simulation.
            aggregator: Optional :class:`~krum.primitives.aggregators.Aggregator`
                subclass overriding the default nearest-neighbor average over the
                ``n - 2f`` closest models. Its ``aggregate`` classmethod is
                called with the received models and a ``pivot``.
            aggregator_kwargs: Extra keyword arguments forwarded to
                ``aggregator.aggregate`` (e.g. ``{"num_closest": ...}``). The
                ``pivot`` is injected per worker; the default aggregator's
                ``num_closest`` is injected automatically.
            byzantine_reach: ``"all"`` to inject every Byzantine model into every
                worker's received set, or ``"sampled"`` to draw responders
                uniformly from all other nodes.
            seed: Optional integer seed for responder sampling.

        Raises:
            ValueError: If a worker count, learning rate, beta, data length, or
                ``byzantine_reach`` is out of range, or an attack is missing
                while ``num_byzantine > 0``.
            TypeError: If ``model`` or ``loss_fn`` has the wrong type, ``attack``
                is not an :class:`~krum.primitives.attacks.Attack` subclass,
                ``aggregator`` is not an
                :class:`~krum.primitives.aggregators.Aggregator` subclass, or
                ``seed`` is not an int or ``None``.
        """
        if byzantine_reach not in ("all", "sampled"):
            raise ValueError(f"Expected byzantine_reach to be 'all' or 'sampled', got {byzantine_reach!r}")

        resolved_aggregator = aggregator or NearestNeighborAverage
        resolved_kwargs = dict(aggregator_kwargs or {})
        if aggregator is None:
            # Each worker mixes over its n - f received models and keeps the
            # n - 2f closest to its own; num_honest - num_byzantine == n - 2f.
            resolved_kwargs.setdefault("num_closest", num_honest - num_byzantine)

        super().__init__(
            model=model,
            data=data,
            loss_fn=loss_fn,
            num_honest=num_honest,
            num_byzantine=num_byzantine,
            learning_rate=learning_rate,
            aggregator=resolved_aggregator,
            beta=beta,
            attack=attack,
            attack_kwargs=attack_kwargs,
            aggregator_kwargs=resolved_kwargs,
            seed=seed,
        )
        self.byzantine_reach = byzantine_reach

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
        permutation = torch.randperm(other_indices.numel(), generator=self.generator)
        return other_indices[permutation[:num_responders].to(device)]

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
        permutation = torch.randperm(other_indices.numel(), generator=self.generator)
        return other_indices[permutation[:num_received].to(device)]
