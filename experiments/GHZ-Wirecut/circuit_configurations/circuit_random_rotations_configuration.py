from circuit_arguments.circuit_argument_interface import CircuitArgumentInterface
from circuit_arguments.random_circuit_rotation_gate import RandomCircuitRotationGate
from circuit_configurations.circuit_configuration_interface import (
    CircuitConfigurationInterface,
)


class RandomCircuitRotationsConfiguration(CircuitConfigurationInterface):
    def __init__(self, numQubits: int, numGates: int):
        self.numQubits = numQubits
        self.numGates = numGates
        self.gates = []
        for i in range(numGates):
            rndRotGate = RandomCircuitRotationGate(self.numQubits)
            self.gates.append(rndRotGate)

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
        string = "Random configuration: "
        for arg in self.gates:
            string += str(arg) + "\n"
        return string
