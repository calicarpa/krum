"""Registry for gradient aggregation rules."""

from __future__ import annotations

from collections.abc import Callable

from .. import tools
from .base import Aggregator, AggregatorSpec, FunctionAggregator


class AggregatorRegistry:
    """Store GAR callables, aliases, metadata, and object adapters."""

    def __init__(self) -> None:
        """Create an empty GAR registry."""
        self.callables: dict[str, Callable] = {}
        self.aliases: dict[str, str] = {}
        self.specs: dict[str, AggregatorSpec] = {}
        self.objects: dict[str, Aggregator] = {}

    def register(
        self,
        spec: AggregatorSpec,
        rule: Callable,
        check: Callable,
    ) -> None:
        """Register a GAR callable with metadata."""
        name = spec.name
        if name in self.callables or name in self.aliases:
            tools.warning(f"Unable to register {name!r} GAR: name already in use")
            return

        self.callables[name] = rule
        self.specs[name] = spec
        self.objects[name] = FunctionAggregator(spec, rule.unchecked, rule.check)
        for alias in spec.aliases:
            if alias in self.callables or alias in self.aliases:
                tools.warning(f"Unable to register {alias!r} alias for {name!r} GAR: name already in use")
                continue
            self.aliases[alias] = name
            self.callables[alias] = rule

    def register_object(self, aggregator: Aggregator, rule: Callable) -> None:
        """Register a GAR object and its callable wrapper."""
        self.register(aggregator.spec, rule, aggregator.check)
        self.objects[aggregator.spec.name] = aggregator

    def get(self, name: str) -> Callable:
        """Return a registered GAR by canonical name or alias."""
        return self.callables[name]

    def get_spec(self, name: str) -> AggregatorSpec:
        """Return GAR metadata by canonical name or alias."""
        canonical = self.aliases.get(name, name)
        return self.specs[canonical]

    def get_object(self, name: str) -> Aggregator:
        """Return a GAR object adapter by canonical name or alias."""
        canonical = self.aliases.get(name, name)
        return self.objects[canonical]
