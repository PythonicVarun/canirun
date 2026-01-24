"""Enums for compatibility levels."""

from enum import Enum


class COMPATIBILITY(int, Enum):
    """Compatibility levels for model execution on hardware."""

    FULL = 1
    PARTIAL = 0
    NONE = -1
