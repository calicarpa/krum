"""Registry for Byzantine attacks."""

from __future__ import annotations

from collections.abc import Callable

from .. import tools
from .base import Attack, AttackSpec, FunctionAttack


class AttackRegistry:
    """Store attack callables, aliases, metadata, and object adapters."""

    def __init__(self) -> None:
        self.callables: dict[str, Callable] = {}
        self.aliases: dict[str, str] = {}
        self.specs: dict[str, AttackSpec] = {}
        self.objects: dict[str, Attack] = {}

    def register(
        self,
        spec: AttackSpec,
        attack: Callable,
        check: Callable,
    ) -> None:
        """Register an attack callable with metadata."""
        name = spec.name
        if name in self.callables or name in self.aliases:
            tools.warning(f"Unable to register {name!r} attack: name already in use")
            return

        self.callables[name] = attack
        self.specs[name] = spec
        self.objects[name] = FunctionAttack(spec, attack.unchecked, attack.check)
        for alias in spec.aliases:
            if alias in self.callables or alias in self.aliases:
                tools.warning(f"Unable to register {alias!r} alias for {name!r} attack: name already in use")
                continue
            self.aliases[alias] = name
            self.callables[alias] = attack

    def register_object(self, attack_object: Attack, attack: Callable) -> None:
        """Register an attack object and its callable wrapper."""
        self.register(attack_object.spec, attack, attack_object.check)
        self.objects[attack_object.spec.name] = attack_object

    def get(self, name: str) -> Callable:
        """Return a registered attack by canonical name or alias."""
        return self.callables[name]

    def get_spec(self, name: str) -> AttackSpec:
        """Return attack metadata by canonical name or alias."""
        canonical = self.aliases.get(name, name)
        return self.specs[canonical]

    def get_object(self, name: str) -> Attack:
        """Return an attack object adapter by canonical name or alias."""
        canonical = self.aliases.get(name, name)
        return self.objects[canonical]
