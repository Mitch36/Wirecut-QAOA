import pennylane as qml
from circuit_configurations.circuit_configuration import CircuitConfiguration
from subcircuit_contribution import SubCircuitContribution
from subcircuit_measurement import SubCircuitMeasurement
from subcircuit_position import SubcircuitPosition


class SubCircuitRandom:
    def __init__(
        self,
        index: int,
        numQubits: int,
        contribution: SubCircuitContribution,
        position: SubcircuitPosition,
        shots: int = 1000,
        config: CircuitConfiguration = None,
    ):
        self.numQubits = numQubits
        self.index = index
        self.contribution = contribution
        self.position = position
        self.shots = shots
        self.config = config
        self.device = qml.device("default.qubit", wires=self.numQubits, shots=shots)

    def __str__(self) -> str:
        return (
            "Subcircuit: "
            + str(self.index)
            + " with "
            + str(self.numQubits)
            + " qubits; has position "
            + str(self.position.name)
            + " and "
            + str(self.shots)
            + " shots and"
            + self.contribution.ToString()
        )
