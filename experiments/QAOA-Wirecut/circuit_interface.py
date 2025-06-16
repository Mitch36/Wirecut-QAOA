from abc import ABC, abstractmethod


class CircuitInterface(ABC):
    """
    Interface for defining the functionality of a quantum circuit.
    """

    @abstractmethod
    def Run(self):
        """
        Abstract method to run the quantum circuit locally using a certain quantum backend (e.g. Pennylane or Qiskit-Aer).
        """
        pass

    @abstractmethod
    def Visualise(self):
        """
        Abstract method to visualize and print the quantum circuit.
        """
        pass