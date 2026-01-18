"""
Flag constants for computed function decorators.

These flags control the behavior of computed functions:
- Serialized: The result will be automatically serialized/persisted
- Input: The value can be permanently set (set)
- Overridable: The value can be temporarily overridden (override)
- Persisted: Combination of Input and Serialized
- Optional: Return NO_VALUE instead of raising exceptions
- CanChange: Indicates the cell can be modified (Input, Overridable, or has inverse)
"""

from typing import Final


class Flags:
    """Bit flags for computed function properties."""

    NONE: Final[int] = 0
    Serialized: Final[int] = 1 << 0    # Result will be serialized
    Input: Final[int] = 1 << 1         # Can be set permanently
    Overridable: Final[int] = 1 << 2   # Can be temporarily overridden
    Optional: Final[int] = 1 << 3      # Return NO_VALUE instead of raising

    # Compound flags
    Persisted: Final[int] = Input | Serialized  # Input + Serialized
    CanChange: Final[int] = Input | Overridable  # Can be modified somehow


# Sentinel value for suppressed errors or missing values
class _NoValue:
    """Sentinel class for representing missing/error values."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NO_VALUE"

    def __bool__(self) -> bool:
        return False


NO_VALUE: Final = _NoValue()


# Export individual flags at module level for convenience
Serialized = Flags.Serialized
Input = Flags.Input
Overridable = Flags.Overridable
Persisted = Flags.Persisted
Optional = Flags.Optional
CanChange = Flags.CanChange
