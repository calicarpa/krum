"""Base types for gradient aggregation rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AggregatorSpec:
    """Metadata describing a gradient aggregation rule."""

    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    paper: str | None = None
    supports_native: bool = False
    upper_bound: Callable[[int, int, int], float] | None = None
    influence: Callable[..., float] | None = None


class Aggregator:
    """Base class for gradient aggregation rules."""

    spec: AggregatorSpec

    def check(self, **kwargs: Any) -> str | None:
        """Validate aggregation parameters."""
        return None

    def aggregate(self, **kwargs: Any) -> torch.Tensor:
        """Aggregate gradients into one gradient."""
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        """Aggregate gradients into one gradient."""
        return self.aggregate(**kwargs)


class FunctionAggregator(Aggregator):
    """Adapter exposing existing function-based GARs through ``Aggregator``."""

    def __init__(
        self,
        spec: AggregatorSpec,
        aggregate: Callable[..., torch.Tensor],
        check: Callable[..., str | None],
    ) -> None:
        """Create an adapter around function-based GAR pieces."""
        self.spec = spec
        self._aggregate = aggregate
        self._check = check

    def check(self, **kwargs: Any) -> str | None:
        """Validate aggregation parameters."""
        return self._check(**kwargs)

    def aggregate(self, **kwargs: Any) -> torch.Tensor:
        """Aggregate gradients into one gradient."""
        return self._aggregate(**kwargs)
