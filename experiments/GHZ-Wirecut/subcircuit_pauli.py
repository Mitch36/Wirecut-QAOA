from math import floor

import pennylane as qml
from circuit_configurations.circuit_configuration import CircuitConfiguration
from subcircuit_contribution import SubCircuitContribution
from subcircuit_measurement import SubCircuitMeasurement
from subcircuit_position import SubcircuitPosition


class SubCircuitPauli:
    def __init__(
        self,
        index: int,
        numQubits: int,
        contribution: SubCircuitContribution,
        position: SubcircuitPosition,
        shotsBudget: int = 1000,
        config: CircuitConfiguration = None,
    ):
        self.numQubits = numQubits
        self.index = index
        self.contribution = contribution
        self.position = position
        self.config = config

        self.shotsBudget = shotsBudget
        self.circuitVariants = self.__getVariants__()
        self.shotsPerVariant = floor(shotsBudget / self.circuitVariants)

        self.device = qml.device("default.qubit", wires=self.numQubits, shots=self.shotsPerVariant)

    def __getVariants__(self) -> int:
        """
        Calculates, based on the subcircuit position, how many variants of the subcircuit are required.
        """
        match self.position:
            case SubcircuitPosition.BEGIN:
                return 3
            case SubcircuitPosition.INTERMEDIATE:
                return 12
            case SubcircuitPosition.END:
                return 4
            case _:
                raise Exception("Invalid parameter, position does not exist")

    def __str__(self) -> str:
        return (
            "Subcircuit: "
            + str(self.index)
            + " with "
            + str(self.numQubits)
            + " qubits; has position "
            + str(self.position.name)
            + " and "
            + str(self.shotsBudget)
            + " shots budget; contribution: "
            + self.contribution.ToString()
        )

    def ToString(self) -> str:
        return self.__str__()

    def Run(self, prepareBasis: chr = "0", measureBasis: chr = "Z") -> SubCircuitMeasurement:
        circuit = qml.QNode(self.__circuit__, self.device)
        prob = circuit(prepareBasis, measureBasis)
        measName = str(self.index) + "_p" + prepareBasis + "_m" + measureBasis
        return SubCircuitMeasurement(self.index, measName, prepareBasis, measureBasis, prob, self.shotsPerVariant)

    def Visualise(self):
        circuit = qml.QNode(self.__circuit__, self.device)
        print(self.ToString())
        print(qml.draw(circuit)("0", "Z"))
        print("\n")

    def GetContribution(self, stateStr: str) -> str:
        if stateStr == None:
            raise Exception("Parameter invalid: stateStr is not of type string")
        if len(stateStr) == 0:
            raise Exception("Parameter invalid: received empty stateStr parameter")

        contribution = self.contribution.ToContribution(stateStr)
        if self.position == SubcircuitPosition.INTERMEDIATE:
            contribution += stateStr[self.contribution.endIndex]
        return contribution

    def __handleConfig__(self, subQubit: int):
        """
        Checks whether the configuration has a specific argument for the current qubit and applies it
        """
        # Calculate the original qubit meant for the argument in the original circuit
        orgQubit = subQubit + self.contribution.startIndex

        # Contribution might overlap; therefore,
        if self.position != SubcircuitPosition.END:
            if orgQubit == self.contribution.endIndex and orgQubit != self.contribution.startIndex:
                return

        if self.config != None:
            args = self.config.Find(orgQubit)
            if len(args) > 0:
                for arg in args:
                    arg.Apply(subQubit)

    def __circuit__(self, prepareBasis: chr = "0", measureBasis: chr = "Z"):
        if self.position == SubcircuitPosition.BEGIN:
            qml.Hadamard(wires=[0])

        match prepareBasis:
            case "0":
                pass  # Prepare qubit in basis: |0> & |1>, no gate added
            case "1":
                qml.PauliX(wires=[0])  # Prepare qubit in basis: |0> & |1>
            case "+":
                qml.Hadamard(wires=[0])  # Prepare qubit in basis: |+>
            case "i":
                qml.Hadamard(wires=[0])  # Prepare qubit in basis |i>
                qml.S(wires=[0])
            case _:
                raise Exception("Invalid parameter, basis does not exist")

        for i in range(self.numQubits):
            self.__handleConfig__(i)  # Check if there is a config on this qubit
            if i != self.numQubits - 1:
                qml.CNOT(wires=[i, i + 1])

        my_array = list(range(self.numQubits))

        match measureBasis:
            case "Z":
                return qml.probs(wires=my_array)  # Measures in basis: |0> & |1>
            case "I":
                return qml.probs(wires=my_array)  # Measures in basis: |0> & |1>
            case "X":
                qml.Hadamard(wires=[1])  # Measures second qubit in basis: |+>
                return qml.probs(wires=my_array)
            case "Y":
                qml.adjoint(qml.S(wires=[1]))  # Inverse S gate or Sdg gate
                qml.Hadamard(wires=[1])  # Measures the second qubit in basis |i>
                return qml.probs(wires=my_array)
            case _:
                raise Exception("Invalid parameter, basis does not exist")
