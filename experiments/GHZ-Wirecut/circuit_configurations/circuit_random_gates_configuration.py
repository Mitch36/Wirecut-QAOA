import random

from circuit_arguments.circuit_argument_interface import CircuitArgumentInterface
from circuit_arguments.circuit_gate_argument import CircuitGateArgument
from circuit_configurations.circuit_configuration_interface import (
    CircuitConfigurationInterface,
)
from circuit_gate import CircuitGate


class RandomCircuitGatesConfiguration(CircuitConfigurationInterface):
    """
    Configuration with random gates on random qubits,  uses CircuitConfiguration as a base class.
    """

    def __init__(self, numQubits: int, numGates: int):
        self.gates = []
        for i in range(numGates):
            gate = random.choice(list(CircuitGate))
            # H gate is not supported
            # The subcircuits are being created before knowing if there is going to be entanglement
            # This makes reconstructing the original probability distribution very difficult.
            while gate == CircuitGate.H:
                gate = random.choice(list(CircuitGate))
            qubit = random.choice(range(numQubits))
            self.gates.append(CircuitGateArgument(qubit, gate))

    def Add(self, arg: CircuitArgumentInterface):
        self.gates.append(arg)

    def Find(self, qubit: int) -> list[CircuitArgumentInterface]:
        """
        Finds all arguments related to a specific qubit.

        Args:
            qubit (int): The qubit to search for.

        Returns:
            list[CircuitArgument]: A list of arguments that affect the qubit.

        Raises:
            Exception: If the qubit is negative.
        """

        if qubit < 0:
            raise ValueError("Parameter invalid: qubit; Qubit cannot be negative")

        result = []
        for arg in self.gates:
            if arg.qubit == qubit:
                result.append(arg)
        return result

    def __str__(self) -> str:
        string = "RandomCircuitGatesConfiguration: "
        for arg in self.gates:
            string += str(arg) + "\n"
        return string
