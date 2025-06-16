from enum import Enum


class SubcircuitPosition(Enum):
    """Enum to specify the position of a subcircuit in a subcircuit-chain."""

    BEGIN = 1
    INTERMEDIATE = 2
    END = 3
