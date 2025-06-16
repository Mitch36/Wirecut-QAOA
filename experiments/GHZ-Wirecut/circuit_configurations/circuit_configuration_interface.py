from abc import ABC, abstractmethod

from circuit_arguments.circuit_argument_interface import CircuitArgumentInterface


class CircuitConfigurationInterface(ABC):
    """
    Interface for defining the arguments of a quantum circuit.
    """

    @abstractmethod
    def Add(self, arg: CircuitArgumentInterface):
        """
        Abstract method to add a circuit argument to the configuration.
        """
        pass

    def Find(self, qubit: int) -> list[CircuitArgumentInterface]:
        """
        Abstract method to find all arguments related to a specific qubit.
        """
        pass
