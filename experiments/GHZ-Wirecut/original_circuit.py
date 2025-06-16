import pennylane as qml
from circuit_configurations.circuit_configuration import CircuitConfiguration


class Circuit:
    def __init__(self, name: str, numQubits: int, shots: int = 1000, config: CircuitConfiguration = None):
        self.numQubits = numQubits
        self.name = name
        self.shots = shots
        self.config = config
        self.device = qml.device("default.qubit", wires=numQubits, shots=shots)

    def __str__(self) -> str:
        return (
            "Original circuit: "
            + self.name
            + " with "
            + str(self.numQubits)
            + " qubits; shots: "
            + str(self.shots)
            + " and \n"
            + str(self.config)
        )

    def ToString(self) -> str:
        return self.__str__()

    def Run(self):
        circuit = qml.QNode(self.__circuit__, self.device)
        return circuit()

    def Visualise(self):
        circuit = qml.QNode(self.__circuit__, self.device)
        print(f"{self.name} GHZ state quantum circuit:")
        print(qml.draw(circuit)())
        print("\n")

    def __circuit__(self):
        qml.Hadamard(wires=[0])

        for i in range(self.numQubits):
            if self.config != None:
                args = self.config.Find(i)
                if len(args) > 0:
                    for arg in args:
                        arg.Apply()
            if i != self.numQubits - 1:
                qml.CNOT(wires=[i, i + 1])

        my_array = list(range(self.numQubits))
        return qml.probs(wires=my_array)
