"""Base types for Byzantine attacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AttackSpec:
    """Metadata describing a Byzantine attack."""

    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    paper: str | None = None
    category: str = "gradient"
    stateful: bool = False
    white_box: bool = False


class Attack:
    """Base class for Byzantine attacks."""

    spec: AttackSpec

    def check(self, **kwargs: Any) -> str | None:
        """Validate attack parameters."""
        return None

    def generate(self, **kwargs: Any) -> list[torch.Tensor]:
        """Generate Byzantine gradients."""
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> list[torch.Tensor]:
        """Generate Byzantine gradients."""
        return self.generate(**kwargs)


class FunctionAttack(Attack):
    """Adapter exposing existing function-based attacks through ``Attack``."""

    def __init__(
        self,
        spec: AttackSpec,
        generate: Callable[..., list[torch.Tensor]],
        check: Callable[..., str | None],
    ) -> None:
        self.spec = spec
        self._generate = generate
        self._check = check

    def check(self, **kwargs: Any) -> str | None:
        """Validate attack parameters."""
        return self._check(**kwargs)

    def generate(self, **kwargs: Any) -> list[torch.Tensor]:
        """Generate Byzantine gradients."""
        return self._generate(**kwargs)
