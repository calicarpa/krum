"""Direction enum for attacks."""

from enum import Enum


class Direction(str, Enum):
    """Positive or negative attack direction."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
