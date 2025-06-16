from enum import Enum


class QuantumChannel(Enum):
    """
    Enum to specify the quantum channel of a subcircuit in a subcircuit-chain.
    Reason: Improve readability and maintainability of the code.

    """

    RANDOM_CLIFFORD = 0
    DEPOLARIZATION = 1
