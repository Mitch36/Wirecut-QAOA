from enum import Enum


class CircuitGate(Enum):
    """Enum to specify the single qubit gates of a circuit argument."""

    H = "H"
    X = "X"
    Y = "Y"
    Z = "Z"
    S = "S"
