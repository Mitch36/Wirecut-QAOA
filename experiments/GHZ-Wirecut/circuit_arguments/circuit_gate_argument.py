import pennylane as qml
from circuit_argument_interface import CircuitArgumentInterface
from circuit_gate import CircuitGate


class CircuitGateArgument(CircuitArgumentInterface):
    def __init__(self, qubit: int, gate: CircuitGate):
        self.qubit = qubit
        self.gate = gate

    def Apply(self, overwriteQubit: int = -1):
        """
        Applies the CircuitArgument, specified by its gate variable, to the qubit.

        Args:
            overwriteQubit (int, optional): The qubit to apply the gate to. Defaults the already specified self.qubit variable.

        Raises:
            Exception: If the gate is not supported or does not exist.
        """

        affectedQubit = self.qubit
        if overwriteQubit != -1:
            affectedQubit = overwriteQubit
        match self.gate:
            case CircuitGate.H:
                qml.Hadamard(wires=[affectedQubit])
            case CircuitGate.X:
                qml.PauliX(wires=[affectedQubit])
            case CircuitGate.Y:
                qml.PauliY(wires=[affectedQubit])
            case CircuitGate.Z:
                qml.PauliZ(wires=[affectedQubit])
            case CircuitGate.S:
                qml.S(wires=[affectedQubit])
            case _:
                raise Exception("Invalid parameter, specified gate variable not supported or does not exist.")

    def __str__(self):
        return f"CircuitGateArgument: Affects qubit: {self.qubit}; applies gate: {self.gate.value}"
