from abc import ABC, abstractmethod


class CircuitArgumentInterface(ABC):
    """
    Interface for defining the arguments of a quantum circuit.
    """

    @abstractmethod
    def Apply(self, overwriteQubit: int = -1):
        """
        Abstract method to apply the circuit argument to a qubit with overwrite functionaility.
        """
        pass
