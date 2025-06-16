from quantum_utils import QuantumUtils as qu


class SubCircuitMeasurement:
    def __init__(self, index: int, name: str, prepareBasis: chr, measureBasis: chr, probabilities: list, shots: int):
        self.index = index
        self.name = name
        self.prepareBasis = prepareBasis
        self.measureBasis = measureBasis
        self.probabilities = probabilities
        self.shots = shots

    def GetProb(self, state: str) -> float:
        if state == None:
            raise Exception("State parameter must have a value")
        if len(self.probabilities) == 0:
            raise Exception("Measurement does not contain any data")
        # Check if out of bounds
        decimal = qu.ToDecimal(state)
        if decimal > len(self.probabilities):
            raise ValueError("State could not be found in probabilities array")

        return self.probabilities[decimal]

    def ToString(self) -> str:
        return (
            str(self.index)
            + " "
            + self.name
            + " prepareBasis: "
            + str(self.prepareBasis)
            + " measureBasis: "
            + str(self.measureBasis)
            + " shots: "
            + str(self.shots)
        )

    def __str__(self) -> str:
        return self.ToString()
