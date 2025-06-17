import math
from math import floor

from circuit_configurations.circuit_configuration import CircuitConfiguration
from original_circuit import Circuit
from quantum_utils.src.quantum_utils import QuantumUtils as qu
from quantum_wire_cutting import QuantumWireCutUtils as wu
from subcircuit_contribution import SubCircuitContribution
from subcircuit_measurement import SubCircuitMeasurement
from subcircuit_pauli import SubCircuitPauli
from subcircuit_position import SubcircuitPosition


class GHZPauliWireCut:
    def __init__(
        self,
        numQubits: int = 3,
        numCuts: int = 1,
        shotsBudget: int = 1000,
        orgShots: int = 1000,
        config: CircuitConfiguration = None,
    ):

        if numQubits < 2:
            raise Exception("Number of qubits must be greater than 1")
        if numCuts < 1:
            raise Exception("Number of cuts must be greater than 0")
        if shotsBudget < 1:
            raise Exception("Number of shots must be greater than 0")

        # Initialize all variables for the GHZExperiment class
        self.numQubits = numQubits
        self.numCuts = numCuts
        self.numSubCircuits = numCuts + 1
        self.numSubCircuitQubits = (numQubits + self.numCuts) / self.numSubCircuits
        self.numCNOT = numQubits - 1

        # Variables used for shots distribution among each subcircuit
        self.orgShots = orgShots
        self.shotsBudget = shotsBudget
        self.subCircuits = []
        self.subCircuitVariants = self.__getTotalVariants__()  # Number of subcircuit variants
        self.shotsPerVariant = floor(self.shotsBudget / self.subCircuitVariants)

        self.measurements = []
        self.config = config

        # Construct original circuit
        self.originalCircuit = Circuit("Original", self.numQubits, self.orgShots, self.config)

        # Calculate subcircuit contribution
        contributions = []
        decimalDump = 0
        contributionIndex = 0

        for i in range(self.numSubCircuits):
            integer_part = math.floor(self.numSubCircuitQubits)
            decimal_part = self.numSubCircuitQubits - integer_part
            decimalDump += decimal_part
            contributions.append(integer_part)

        decimalDump = round(decimalDump)

        for i in range(int(decimalDump)):
            contributions[i] += 1

        # Construct Subcircuit objects
        for i in range(self.numSubCircuits):
            contribution = None
            if i == 0:
                position = SubcircuitPosition.BEGIN
                contribution = SubCircuitContribution(contributionIndex, contributions[i] - 1, self.numQubits)

            elif i == len(contributions) - 1:
                position = SubcircuitPosition.END
                contribution = SubCircuitContribution(contributionIndex, contributions[i], self.numQubits)
            else:
                position = SubcircuitPosition.INTERMEDIATE
                contribution = SubCircuitContribution(contributionIndex, contributions[i] - 1, self.numQubits)

            contributionIndex += contributions[i] - 1
            self.subCircuits.append(
                SubCircuitPauli(
                    i,
                    contributions[i],
                    contribution,
                    position,
                    self.__getShotsBasedOnPosition__(position),
                    self.config,
                )
            )

        # print(
        #     f"{self.numCNOT} CNOT gates are required to make the GHZ quantum circuit (Pauli based) with {self.numQubits} qubits"
        # )
        # print(
        #     f"For {self.numCuts} number of cuts are {self.numSubCircuits} subcircuits required with each having {self.numSubCircuitQubits} qubits"
        # )

    def __getTotalVariants__(self) -> int:
        counter = 7  # BEGIN is 3 and END is 4
        for count in range(2, self.numSubCircuits):
            counter += 12
        self.subCircuitVariants = counter
        return counter

    def __getShotsBasedOnPosition__(self, position: SubcircuitPosition) -> int:
        """
        Returns the number of shots based on the position of the subcircuit
        """
        match position:
            case SubcircuitPosition.BEGIN:
                return 3 * self.shotsPerVariant
            case SubcircuitPosition.INTERMEDIATE:
                return 12 * self.shotsPerVariant
            case SubcircuitPosition.END:
                return 4 * self.shotsPerVariant
            case _:
                raise Exception("Invalid parameter, position does not exist")

    def RunOriginialCircuit(self):
        return self.originalCircuit.Run()

    def DoSubCircuitMeasurements(self) -> int:
        if len(self.subCircuits) == 0:
            raise Exception("No subcircuits have been created")

        # Loop through all subcircuits and perform measurements based on their position within the subcircuit chain
        # BEGIN -> INTERMEDIATE -> ... -> END

        self.measurements = []

        for subcircuit in self.subCircuits:
            match subcircuit.position:
                case SubcircuitPosition.BEGIN:
                    self.measurements.append(subcircuit.Run("0", "X"))
                    self.measurements.append(subcircuit.Run("0", "Y"))
                    self.measurements.append(subcircuit.Run("0", "Z"))
                case SubcircuitPosition.INTERMEDIATE:
                    for prepareBasis in ["0", "1", "+", "i"]:
                        self.measurements.append(subcircuit.Run(prepareBasis, "X"))
                        self.measurements.append(subcircuit.Run(prepareBasis, "Y"))
                        self.measurements.append(subcircuit.Run(prepareBasis, "Z"))
                case SubcircuitPosition.END:
                    self.measurements.append(subcircuit.Run("0", "Z"))
                    self.measurements.append(subcircuit.Run("1", "Z"))
                    self.measurements.append(subcircuit.Run("+", "Z"))
                    self.measurements.append(subcircuit.Run("i", "Z"))
                case _:
                    raise Exception("Invalid subcircuit position")

        return self.shotsBudget

    def PrintMeasurements(self):
        """
        Prints all measurements in the measurements list
        """
        print("Total sub-circuit shots used for measurements: ", self.shotsBudget)
        for entry in self.measurements:
            print(entry.ToString())

    def GetMeasurement(self, index: int, prepareBasis: chr, measureBasis: chr) -> SubCircuitMeasurement:
        """
        Gets a measurement from the measurements list based on the index, prepareBasis and measureBasis.

        Args:
            index (int): The index of the measurement
            prepareBasis (chr): The prepare basis
            measureBasis (chr): The measure basis

        Returns:
            SubCircuitMeasurement: The measurement data

        Raises:
            ValueError: If the measurement data does not exist with the given parameters
        """
        for data in self.measurements:
            if data.index == index and data.prepareBasis == prepareBasis and data.measureBasis == measureBasis:
                return data
        raise ValueError("Measurement data does not exist with given parameters")

    def __calculateP1__(self, stateStr: str, subcircuit: SubCircuitPauli, prepareBasis: chr) -> list:

        if prepareBasis != "0" and prepareBasis != "1" and prepareBasis != "+" and prepareBasis != "i":
            raise ValueError("Parameter invalid: 'prepareBasis'; must be 1, 0, + or i")

        measX = self.GetMeasurement(subcircuit.index, prepareBasis, "X")
        measY = self.GetMeasurement(subcircuit.index, prepareBasis, "Y")
        measZ = self.GetMeasurement(subcircuit.index, prepareBasis, "Z")

        # Get stateStr according to the subcircuit contribution
        stateStr = subcircuit.contribution.ToContribution(stateStr)
        excludeStr = stateStr

        stateStr += "0"
        excludeStr += "1"

        p1 = []
        p1.append(
            measZ.GetProb(stateStr) + measZ.GetProb(excludeStr) + measZ.GetProb(stateStr) - measZ.GetProb(excludeStr)
        )
        p1.append(
            measZ.GetProb(stateStr) + measZ.GetProb(excludeStr) - measZ.GetProb(stateStr) + measZ.GetProb(excludeStr)
        )
        p1.append(measX.GetProb(stateStr) - measX.GetProb(excludeStr))
        p1.append(measY.GetProb(stateStr) - measY.GetProb(excludeStr))

        return p1

    def __calculateP2__(self, stateStr: str, subcircuit: SubCircuitPauli) -> list:

        meas0 = self.GetMeasurement(subcircuit.index, "0", "Z")
        meas1 = self.GetMeasurement(subcircuit.index, "1", "Z")
        measPlus = self.GetMeasurement(subcircuit.index, "+", "Z")
        measi = self.GetMeasurement(subcircuit.index, "i", "Z")

        p2 = []

        # Get stateStr according to the subcircuit contribution
        stateStr = subcircuit.GetContribution(stateStr)  # Returns 01

        # Check whether the contribution length is less than the stateStr length, if so plus the probability chances of the overflow qubits, same trick as applied in p1
        # Example: contribution is 0, stateStr is 01, we only want the prob of 0X, therefore 00p2 - 01p2
        diff = len(stateStr) - subcircuit.contribution.length
        if diff > 0:
            stateStr0 = stateStr[0 : subcircuit.contribution.length]
            stateStr1 = stateStr[0 : subcircuit.contribution.length]
            for i in range(diff):
                stateStr0 += "0"
                stateStr1 += "1"
                p2.append(meas0.GetProb(stateStr0) + meas0.GetProb(stateStr1))
                p2.append(meas1.GetProb(stateStr0) + meas1.GetProb(stateStr1))
                p2.append(
                    (2 * (measPlus.GetProb(stateStr0)) - meas0.GetProb(stateStr0) - meas1.GetProb(stateStr0))
                    + (2 * (measPlus.GetProb(stateStr1)) - meas0.GetProb(stateStr1) - meas1.GetProb(stateStr1))
                )
                p2.append(
                    (2 * (measi.GetProb(stateStr0)) - meas0.GetProb(stateStr0) - meas1.GetProb(stateStr0))
                    + (2 * (measi.GetProb(stateStr1)) - meas0.GetProb(stateStr1) - meas1.GetProb(stateStr1))
                )

        else:
            p2.append(meas0.GetProb(stateStr))
            p2.append(meas1.GetProb(stateStr))
            p2.append(2 * (measPlus.GetProb(stateStr)) - meas0.GetProb(stateStr) - meas1.GetProb(stateStr))
            p2.append(2 * (measi.GetProb(stateStr)) - meas0.GetProb(stateStr) - meas1.GetProb(stateStr))

        return p2

    def ConstructProbability(self, stateStr: str) -> float:
        # First clean up stateStr by removing '|' and '>'
        index = stateStr.find("|")
        if index != -1:
            stateStr = stateStr[:index] + stateStr[index + 1 :]
        index = stateStr.find(">")
        if index != -1:
            stateStr = stateStr[:index] + stateStr[index + 1 :]
        if len(stateStr) != self.numQubits:
            raise ValueError(
                f"State parameter must contain the same number of qubits ({self.numQubits}) as characters ({len(stateStr)})"
            )

        # Evualate each cut independently
        # First cut, it can be safely assumed that for the first subcircuit
        # all qubits are initliased with |0>; therefore the reconstruction of the probabilities
        # is shortened
        subA = self.subCircuits[0]
        subB = self.subCircuits[1]

        p1 = self.__calculateP1__(stateStr, subA, "0")
        p2 = self.__calculateP2__(stateStr, subB)

        segments = []
        result = wu.KroneckerProductSum(p1, p2)[0][0]
        if result < 0:
            result = 0
        segments.append(result)

        # If there is only a single cut, early return, no further action required.
        if self.numCuts == 1:
            return segments[0]

        for cutIndex in range(1, self.numCuts):
            # Loop through every cut and calculate each probability
            p1, p2 = [], None
            p1.append(self.__calculateP1__(stateStr, self.subCircuits[cutIndex], "0"))
            p1.append(self.__calculateP1__(stateStr, self.subCircuits[cutIndex], "1"))
            p1.append(self.__calculateP1__(stateStr, self.subCircuits[cutIndex], "+"))
            p1.append(self.__calculateP1__(stateStr, self.subCircuits[cutIndex], "i"))

            p2 = self.__calculateP2__(stateStr, self.subCircuits[cutIndex + 1])

            sum = 0
            for i in range(4):
                sum += wu.KroneckerProductSum(p1[i], p2)[0][0]
            if sum < 0:
                sum = 0
            segments.append(sum / 2)

        sum = segments[0]
        for i in range(1, len(segments)):
            sum = sum * segments[i]

        return sum

    def ConstructFullProbabilites(self) -> list:
        result = []
        for i in range(2**self.numQubits):
            state = qu.ToBinary(self.numQubits, i)
            result.append(self.ConstructProbability(state))
        return qu.Normalize(wu.ComplexToNormal(result))
