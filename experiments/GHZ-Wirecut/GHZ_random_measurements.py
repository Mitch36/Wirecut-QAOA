from abc import abstractmethod
from math import floor
from typing import Tuple

import numpy as np
import pennylane as qml
from circuit_configurations.circuit_configuration import CircuitConfiguration
from original_circuit import Circuit
from quantum_channel import QuantumChannel
from quantum_utils import QuantumUtils as qu
from quantum_wire_cutting import QuantumWireCutUtils as qwc
from subcircuit_contribution import SubCircuitContribution
from subcircuit_position import SubcircuitPosition
from subcircuit_random import SubCircuitRandom


class GHZRandomWireCut:
    def __init__(
        self,
        numQubits: int = 3,
        numCuts: int = 1,
        shotsBudget: int = 100,
        shotIterations: int = 10,
        useQuantumTeleportation: bool = True,
        useRandomRotation: bool = False,
        config: CircuitConfiguration = None,
    ):

        if numQubits < 2:
            raise Exception("Number of qubits must be greater than 1")
        if numCuts < 1:
            raise Exception("Number of cuts must be greater than 0")

        # Initialize all variables for the GHZExperiment class
        self.numQubits = numQubits
        self.numClassicBits = numQubits

        # For measure and prepare channel we also need another classic bit
        measPreChannel = numCuts
        if useQuantumTeleportation:
            measPreChannel *= 2
        self.numClassicBits += measPreChannel

        self.numCuts = numCuts
        self.numSubCircuits = numCuts + 1
        self.numSubCircuitQubits = (numQubits + self.numCuts) / self.numSubCircuits
        self.numCNOT = numQubits - 1
        self.shotsBudget = shotsBudget
        self.subIterations = shotIterations
        self.shotsPerIteration = round(shotsBudget / self.subIterations)
        self.orgShots = 1000

        # Additional flags
        self.useQuantumTeleportation = useQuantumTeleportation
        self.useRandomRotation = useRandomRotation


        self.subCircuits = []
        self.measurements = []
        self.config = config

        # Construct original circuit
        self.originalCircuit = Circuit("Original", self.numQubits, self.orgShots, self.config)

        # Calculate subcircuit contribution
        contributions = []
        decimalDump = 0
        contributionIndex = 0

        for i in range(self.numSubCircuits):
            integer_part = floor(self.numSubCircuitQubits)
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
                SubCircuitRandom(i, contributions[i], contribution, position, self.shotsPerIteration, self.config)
            )

        # print(
        #     f"{self.numCNOT} CNOT gates are required to make the GHZ quantum circuit (Random measurements) with {self.numQubits} qubits"
        # )
        # print(
        #     f"For {self.numCuts} number of cuts are {self.numSubCircuits} subcircuits required with each having {self.numSubCircuitQubits} qubits"
        # )

    def __handleConfig__(self, subcircuit: SubCircuitRandom, qubit: int):
        """
        Helpder function which handles the configuration of the subcircuit based on the config property
        """
        if qubit < 0:
            raise Exception("Qubit cannot be negative")

        orgQubit = qubit + subcircuit.contribution.startIndex

        if not (subcircuit.contribution.startIndex <= orgQubit <= subcircuit.contribution.endIndex):
            return  # Qubit is not in the subcircuit, early return
        
        # Due the overlap of subcircuit objects, check if the qubit is last in the circuit and position is either BEGIN or INTERMEDIATE
        if orgQubit == subcircuit.contribution.endIndex and (subcircuit.position == SubcircuitPosition.BEGIN or subcircuit.position == SubcircuitPosition.INTERMEDIATE):
            return
        
        gates = []
        if self.config != None:
            gates = self.config.Find(orgQubit)
            if len(gates) == 0:
                return  # No gates found for this qubit, early return
            
        # Translate the original qubit parameter to the subcircuit qubit
        subcircuitNumQubits = subcircuit.numQubits

        orgQubit = qubit  # Copy the original qubit
        while qubit > subcircuitNumQubits:
            qubit -= subcircuitNumQubits

        # We now have the subcircuit qubit on which we need to apply the gates
        for gate in gates:
            match subcircuit.position:
                case SubcircuitPosition.BEGIN | SubcircuitPosition.INTERMEDIATE:
                    if orgQubit != subcircuit.contribution.endIndex:
                        gate.Apply(qubit)
                case SubcircuitPosition.END:
                    gate.Apply(qubit)
                case _:
                    raise Exception("Invalid subcircuit position")

    def SubCircuit(self, qChannel: QuantumChannel):

        # Create a list to store the measurements
        measurements = []
        quantumTelData = [0, 0]  # First bit represents measurement in the computational basis, second bit represents measurement hadamard basis
        rndCliff = None

        if qChannel == QuantumChannel.RANDOM_CLIFFORD:  # (Measure and Prepare with 2-Design) (Channel 0)
            for subcircuit in self.subCircuits:
                # Do subcircuit logic
                if subcircuit.position == SubcircuitPosition.BEGIN:
                    qml.Hadamard(wires=[0])
                else:
                    # Qubit must be initialized according to the quantum teleportation data
                    qml.cond(quantumTelData[0], lambda: qml.PauliX(wires=[0]))()
                    if self.useQuantumTeleportation:
                        qml.cond(quantumTelData[1], lambda: qml.PauliZ(wires=[0]))()

                    # Peform random Clifford circuit
                    rndCliff = qwc.WireMove(rndCliff, 0)
                    qwc.ApplyCircuit(rndCliff, adjoint=False)

                # Do more subcircuit logic to create GHZ circuit
                for qubit in range(subcircuit.numQubits):
                    self.__handleConfig__(subcircuit, qubit)  # Check if there is a config on this qubit
                    if not qubit == subcircuit.numQubits - 1:
                        qml.CNOT(wires=[qubit, qubit + 1])

                # Adjointed random Clifford circuit (gets executed immediately after function call)
                if subcircuit.position != SubcircuitPosition.END:
                    if self.useRandomRotation:
                        # rndCliff = qwc.RandomRotationGate(subcircuit.numQubits - 1)
                        rndCliff = qwc.RandomRotations(subcircuit.numQubits - 1)
                    else:
                        rndCliff = qwc.RandomCliffordCircuit(wires=[subcircuit.numQubits - 1], depth=1)
                    qwc.ApplyCircuit(rndCliff, adjoint=True)

                # Measure the contribution qubits based on the subcircuit position
                match subcircuit.position:
                    case SubcircuitPosition.BEGIN | SubcircuitPosition.INTERMEDIATE:
                        for qubit in range(subcircuit.numQubits - 1):
                            measurements.append(qml.measure(wires=[qubit], reset=True))
                    case SubcircuitPosition.END:
                        for qubit in range(subcircuit.numQubits):
                            measurements.append(qml.measure(wires=[qubit], reset=True))
                        return qml.counts(measurements)

                # Reached last qubit, prepare quantum teleportation circuit
                if self.useQuantumTeleportation:
                    qml.CNOT(wires=[subcircuit.numQubits - 1, subcircuit.numQubits - 2])
                    qml.Hadamard(wires=[subcircuit.numQubits - 2])

                    quantumTelData[0] = qml.measure(
                        wires=[subcircuit.numQubits - 1], reset=True
                    )  # First bit is computational basis measurement
                    quantumTelData[1] = qml.measure(
                        wires=[subcircuit.numQubits - 2], reset=True
                    )  # Second bit is hadamard basis measurement
                else:
                    quantumTelData[0] = qml.measure(
                        wires=[subcircuit.numQubits - 1], reset=True
                    )  # First bit is computational basis measurement

        elif qChannel == QuantumChannel.DEPOLARIZATION:  # Depolarization channel (Channel 1)

            # The cut qubit needs to be traced out, therefore destroying its measurement in the first subcircuit

            for subcircuit in self.subCircuits:

                # First do GHZ quantum circuit logic
                if subcircuit.position == SubcircuitPosition.BEGIN:
                    qml.Hadamard(wires=[0])

                for qubit in range(subcircuit.numQubits):
                    self.__handleConfig__(subcircuit, qubit)  # Check if there is a config on this qubit
                    if not qubit == subcircuit.numQubits - 1:
                        qml.CNOT(wires=[qubit, qubit + 1])

                # Second, do measurement logic
                for qubit in range(subcircuit.numQubits):
                    if qubit == subcircuit.numQubits - 1 and subcircuit.position != SubcircuitPosition.END:
                        # Destroy encoded information by doing a measurement
                        qml.measure(wires=qubit, reset=True)
                    else:
                        measurements.append(qml.measure(wires=[qubit], reset=True))
                if subcircuit.position != SubcircuitPosition.END:
                    # Prepare random basis state e.g. |0>, |1>, |+>, |+i>
                    qwc.PrepareRandomBasis(wires=[0])

            return qml.counts(measurements)

        else:
            raise Exception("Invalid quantum channel")
        # Return the counts of the different classical bitstring obtained by all measurements

        return qml.counts(measurements)

    def VisualiseSubcircuit(self, quantumChannel: QuantumChannel):
        """
        Visualise the subcircuit for the given quantum channel

        Args:
            quantumChannel (QuantumChannel): The quantum channel to visualise. Can be either random_clifford or depolarization.
        """
        total = self.subCircuits[0].numQubits + self.numClassicBits
        device = qml.device("default.mixed", wires=total, shots=self.shotsPerIteration)
        circuit = qml.QNode(self.SubCircuit, device)
        print(qml.draw(qnode=circuit, max_length=250)(quantumChannel))
        print("\n")

    def Run(self, printResults: bool = False) -> Tuple[list[int], list[int]]:
        """
        Run the GHZ experiment with random measurements for both quantum channels

        Returns:
            (Tuple[list[int], list[int]]): The results of the two quantum channels
        """

        shotsDistri = self.GetProbQuantumChannels()
        if printResults:
            print("Number of shots for channel 1: ", shotsDistri[1])
            print("Number of shots for channel 0: ", shotsDistri[0])

        measurements0 = []
        measurements1 = []

        if printResults:
            print("Running subcircuit: ", self.subIterations, " times each channel 0 having: ", shotsDistri[0], " shots and channel 1 having: ", shotsDistri[1], " shots")

        for i in range(self.subIterations):
            total = self.subCircuits[0].numQubits + self.numClassicBits
            dev = qml.device("lightning.qubit", wires=total, shots=shotsDistri[0])
            measurements0.append(qml.QNode(self.SubCircuit, dev, mcm_method="one-shot")(QuantumChannel.RANDOM_CLIFFORD))

            dev = qml.device("lightning.qubit", wires=total, shots=shotsDistri[1])
            measurements1.append(qml.QNode(self.SubCircuit, dev, mcm_method="one-shot")(QuantumChannel.DEPOLARIZATION))


        # Summarize results of all subcircuit iterations
        clnChannel0 = []
        clnChannel1 = []

        for bitStr in range(2**self.numQubits):
            clnChannel0.append(0)
            clnChannel1.append(0)
            for mea0, mea1 in zip(measurements0, measurements1):
                val = mea0.get(qu.ToBinary(self.numQubits, bitStr), None)
                if val is not None:
                    clnChannel0[bitStr] += val.sum()
                val = mea1.get(qu.ToBinary(self.numQubits, bitStr), None)
                if val is not None:
                    clnChannel1[bitStr] += val.sum()
        if printResults:
            print("Results of channel 0: ", clnChannel0)
            print("Results of channel 1: ", clnChannel1)

        return clnChannel0, clnChannel1

    @staticmethod
    def MergeAndNormalize(channel0: list, channel1: list) -> list:
        """
        Merge the results of the two quantum channels and scale them

        Args:
            channel0 (list): The results of the first quantum channel
            channel1 (list): The results of the second quantum channel

        Returns:
            list: The merged and scaled results
        """
        merged = []
        d = 2**1
        for i in range(len(channel0)):
            # Apply formula: X = (d + 1)Ψ0(X) - dΨ1(X) from: https://pennylane.ai/qml/demos/tutorial_quantum_circuit_cutting
            result = (d + 1) * channel0[i] - d * channel1[i]
            # Probability can not be negative
            if result < 0:
                result = 0
            merged.append(result)
        return qu.Normalize(merged)

    def GetProbQuantumChannels(self) -> Tuple[int, int]:
        # Apply formula: d = 2^k; k = cut qubits; source: https://pennylane.ai/qml/demos/tutorial_quantum_circuit_cutting
        numCutQbits = 2**1
        chnProb0 = (numCutQbits + 1) / (2 * numCutQbits + 1)
        chnProb1 = numCutQbits / (2 * numCutQbits + 1)

        return round(chnProb0 * self.shotsPerIteration), round(chnProb1 * self.shotsPerIteration)
