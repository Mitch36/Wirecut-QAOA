# Internal libraries
from classical_computer import ClassicalComputer
from experiment_configuration import ExperimentConfiguration as ExpConf
from graph import Graph
from experiment_result import ExperimentResult
from quantum_backend import QuantumBackEnd
from quantum_wire_cutting import QuantumWireCutUtils as qwu
from quantum_utils import QuantumUtils as qu
from quantum_channel import QuantumChannel
from qaoa_utils import QaoaUtils
from classical_computer import ClassicalComputer
from noise_manager import NoiseManager

# IBM Qiskit imports
from qiskit import QuantumCircuit
from qiskit_qasm3_import import parse
from qiskit_ibm_runtime import QiskitRuntimeService # For fetching the noise model of IBM_brisbane and IBM_sherbrooke
from qiskit_aer import AerSimulator # Noisy simulator
from qiskit_aer.noise import NoiseModel # For fetching the noise model of IBM_brisbane and IBM_sherbrooke

# External libraries
import pennylane as qml
from enum import Enum

from scipy.optimize import minimize as MinimizeScipy # Used in classical optimization loop
from spsa import minimize as MinimizeSPSA # Used in classical optimization loop

from copy import deepcopy # Used for computing the edge layout on the cut circuit


class WireCutChannel(Enum):
    RANDOM_CLIFFORD = 0
    DEPOLARIZATION = 1

class WireCutCircuit:
    def __init__(self, graph: Graph, config: ExpConf):
        self.graph: Graph = graph
        self.config: ExpConf = deepcopy(config)
        self.numQaoaLayers: int = config.Get("numQaoaLayers")

        self.numWires: int = self.__computeMinimumNumberOfQubits__()
        self.wires: list[int] = list(range(1, self.numWires + 1))
        self.numCuts = len(self.graph.rows) - 2 # Based on if three rows exist, only one cut can be pleased, for every addtional row, one cut can be placed.

        # Update required qubits parameter in the configuration
        self.config.config["numQubits"] = self.numWires # The number of qubits we need to fit the graph on the wire cut circuit
        self.config.config["numCuts"] = self.numCuts

        # Calculate the number of qubits we are simulating
        self.numOriginalQubits: int = len(graph.graph.nodes) # The number of original qubits we try to simulate
        self.config.config["numOriginalQubits"] = self.numOriginalQubits

        # Used for converting to OPENQASM 3.0
        self.classicalComputer = ClassicalComputer()

        self.shotsBudget: int = config.Get("shotsBudgetPerQuantumCircuit")

        # Before this was set to 100
        # Instead of increasing the subcircuit variants, we increase the number of shots per circuit variant.

        # Classical optimization bookeeping
        self.optimalAvgCost: float = 0.0
        self.optimalParameters: list = []
        self.costHistory: list[float] = []
        self.hasOptimizedParameters: bool = False

        # Bookeeping for the construction of the quantum circuit
        self.wireCutPartitions: list[dict] = self.__ComputeWireCutPartition__()
        self.reversedWireCutPartitions: list[dict] = self.__ReverseWireCutPartitions__()
        self.numCutQubits: int = self.__computeCutQubits__()


    def __computeMinimumNumberOfQubits__(self) -> int:
        # example input is rows = [3, 2, 3]
    
        # compare each neighbouring two layers with each other, to find the minimum number of qubits required to facilitate both layers
        # for example, if we have 3 qubits in the first layer and 2 in the second layer, we need at least 5 qubits to facilitate both layers

        greatestPair: int = 0
        for index in range(len(self.graph.rows)):
            
            if index == len(self.graph.rows) - 1:
                # Reached end of the list, break
                break

            # Calculate new pair
            newPair: int = self.graph.rows[index] + self.graph.rows[index + 1]
            # Check if new pair is greater than the greatest pair
            if newPair > greatestPair:
                greatestPair = newPair
                # Update the number of qubits needed
                self.numWires = newPair

        return greatestPair
    
    def __ComputeWireCutPartition__(self) -> list[dict]:
        rows = deepcopy(self.graph.rows)
        rows.reverse()

        # Convert rows to nodes
        nodeGroups = []; nodeIndex: int = 0
        for row in rows:
            newGroup = []
            for node in range(row):
                nodeIndex += 1
                newGroup.append(nodeIndex)
            nodeGroups.append(newGroup)

        # Convert nodes to groups representing wirecutparts
        wireCutParts: list = []
        for groupIndex in range(len(nodeGroups)-1):
            groupA = nodeGroups[groupIndex]
            groupB = nodeGroups[groupIndex + 1]
            # take first index of groupA and first digit of groupB
            wireCutPart = [groupA[0], groupB[-1]]
            wireCutParts.append(wireCutPart)

        # Fill empty node spaces e.g. [1, 3] -> [1, 2, 3]
        for index, part in enumerate(wireCutParts):
            part = list(range(part[0], part[1] + 1))
            wireCutParts[index] = part

        # Create dictionary for each wirecut part which nodes contribute to the classical bitstring and which are 
        # measure and prepare channels
        wireCutDictList: list[dict] = []
        for counter in range(len(wireCutParts)):
            # Last item reached, all nodes (qubits) contribute to the final bitstring
            partA = wireCutParts[counter]
            if counter == len(wireCutParts) - 1:
                wireCutDictList.append({"contribution": partA, "measure-prepare": [], "all": partA})
                break 
            
            partB = wireCutParts[counter + 1]
            measurePrepareList: list[int] = []; contributionList: list[int] = []

            for node in partA:
                if node in partB:
                    # Node also exists in partB (the next layer), add to measure and prepare
                    measurePrepareList.append(node)
                else:
                    contributionList.append(node)

            wireCutDictList.append({"contribution": contributionList, "measure-prepare": measurePrepareList, "all": partA})
        return wireCutDictList
    
    def __ReverseWireCutPartitions__(self) -> list[dict]:
        # Example input could be:
        # [{'contribution': [1], 'measure-prepare': [2], 'all': [1, 2]}, {'contribution': [2, 3], 'measure-prepare': [], 'all': [2, 3]}]

        partitions = deepcopy(self.wireCutPartitions)

        for partition in partitions:
            for key, wiresList in partition.items():
                reversedOrder :list[int] = []
                
                # Create a reversed array of the wires
                # e.g. [1, 2, 3] -> [3, 2, 1]
                reversedArray = list(reversed(list(range(1, partitions[-1]["all"][-1] + 1)))) # Get the highest possible wire

                for wire in wiresList:
                    # Get the index of the wire in the reversed array
                    reversedWire = reversedArray.index(wire) + 1
                    reversedOrder.append(reversedWire)
                partition[key] = reversedOrder
        
        return partitions
    
    def MapNodeReverse(self, wire: int, partitionIndex: int) -> int:
        partitionMpWires: int = len(self.reversedWireCutPartitions[partitionIndex]["measure-prepare"])
        allPartitionNodes: list[int] = self.reversedWireCutPartitions[partitionIndex]["all"]

        shift: int = 0

        if partitionIndex == 0:
            # First partition, 
            shift = allPartitionNodes[-1] - 1 # Takes the largest wire in the partition, take your own wire into account (-1).
            return wire - shift

        if partitionMpWires == 0:
            # Last partition reached, already mapped
            return wire

        # This is a middle part, somewhere between the first and last partition
        # We do not have to take into account the first partition mp qubits, therefore 1:
        for part in self.reversedWireCutPartitions[partitionIndex:]:
            shift += len(part["measure-prepare"])
        
        return wire - shift -1 # It is unsure why -1 works; further investigatiion is needed.
    
    def MapNode(self, wire: int, partitionIndex: int) -> int:
        partitionMpWires: int = len(self.wireCutPartitions[partitionIndex]["measure-prepare"])

        shift: int = 0

        if partitionIndex == 0:
            # First partition, already mapped
            return wire
        
        if partitionMpWires == 0:
            # Last partition reached
            allPartitionNodes: list[int] = self.wireCutPartitions[partitionIndex]["all"]
            shift: int = allPartitionNodes[0] - 1
            return wire - shift 
        
        # This is a middle part, shifting the node becomes more complicated. 
        # We do not have to take into account the first partition mp qubits, therefore 1:
        for part in self.wireCutPartitions[1:partitionIndex + 1]:
            shift += len(part["measure-prepare"])
        
        return wire - shift
    
    def MapNodes(self, wires: list[int], partitionIndex: int) -> list[int]:

        newWires: list[int] = []
        for wire in wires:
            newWires.append(self.MapNode(wire, partitionIndex))
        return newWires
    
    def MapNodesReverse(self, wires: list[int], partitionIndex: int) -> list[int]:
        newWires: list[int] = []
        for wire in wires:
            newWires.append(self.MapNodeReverse(wire, partitionIndex))
        return newWires

    def MapEdge(self, edge: tuple[int, int], partitionIndex: int) -> tuple[int, int]:
        """
        Maps the edge to the correct index in the circuit.
        """
        return (self.MapNode(edge[0], partitionIndex), self.MapNode(edge[1], partitionIndex))
            
    def MapEdgeReverse(self, edge: tuple[int, int], partitionIndex: int) -> tuple[int, int]:
        """
        Maps the edge to the correct index in the circuit.
        """
        return (self.MapNodeReverse(edge[1], partitionIndex), self.MapNodeReverse(edge[0], partitionIndex))

    def DepolarizationQuantumCircuit(self, gamma: list[float], beta: list[float], layer: int) -> dict:
        classicBitRegister: dict = {}
        nodeInHadamardBasisCounter: int = 0
        
        for wireCutPartIndex, wireCutPart in enumerate(self.wireCutPartitions):
            # 0. Get the specified wire cut partition information
            wireCutPartAll = wireCutPart["all"] # Contains all nodes
            wireCutPartContribution = wireCutPart["contribution"] # Contains all nodes which measurement contributes to the final bitstring
            wireCutPartMeasurePrepare = wireCutPart["measure-prepare"] # Contains all nodes which measurement does not contribute to the final bitstring

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)
            
            # 0. Prepare the qubit in random basis state e.g. (|0>, |1>, |+>, |+i>) because of the previous cut
            if wireCutPartIndex != 0:
                counter: int = 1
                for wire in self.wireCutPartitions[wireCutPartIndex - 1]["measure-prepare"]:
                    qwu.PrepareRandomBasis(wires=counter)
                    counter += 1

            # 1. Apply the Hadamard gates to the new qubits specific wirecutpart
            # If this is the first partition and not the first layer, no Hadamard gates have to be added
            if wireCutPartIndex == 0 and layer != 0:
                for wire in wireCutPartAll:
                    if wire > nodeInHadamardBasisCounter:
                        nodeInHadamardBasisCounter += 1
            else:
                for wire in wireCutPartAll:
                    if wire > nodeInHadamardBasisCounter:
                        qml.Hadamard(wires=self.MapNode(wire, wireCutPartIndex))
                        nodeInHadamardBasisCounter += 1

            
            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 2. Apply cost circuit
            allMin: int = min(wireCutPart["all"]); allMax: int = max(wireCutPart["all"])

            for edge in self.graph.graph.edges:
                if edge[0] >= allMin and edge[1] <= allMax:
                    # Apply the cost circuit
                    qml.IsingZZ(gamma[layer] * 2, wires=self.MapEdge(edge, wireCutPartIndex))

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 3. Appy mixer circuit
            for mappedWire in self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex):
                # Apply the RX gate to the qubit
                qml.RX(beta[layer] * 2, wires=mappedWire)

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 4. Apply the depolarization channel on the measure prepare qubits
            for mappedWire in self.MapNodes(wireCutPartMeasurePrepare, partitionIndex=wireCutPartIndex):
                # 4.1. Destroy encoded information on the qubit using a measurement
                if self.backend == QuantumBackEnd.PENNYLANE:
                    qml.measure(wires=mappedWire, reset=False)
                else:
                    self.classicalComputer.Measure(wire=mappedWire, reset=False)
                
            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 5. If this is not the last layer, apply a depolarization channel on the contribution wires
            if not layer == self.numQaoaLayers - 1 and not wireCutPartIndex == len(self.wireCutPartitions) - 1:
                # Apply the depolarization channel on the contribution wires
                for mappedWire in self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex):
                    # 5.1. Destroy encoded information on the qubit using a measurement
                    if self.backend == QuantumBackEnd.PENNYLANE:
                        qml.measure(wires=mappedWire, reset=False)
                    else:
                        self.classicalComputer.Measure(wire=mappedWire, reset=True)
                    
                    # 5.2. Prepare the qubit in random basis state e.g. (|0>, |1>, |+>, |+i>) for the next part
                    raise NotImplementedError("Not sure if this function is still being called, change if this is the case.")
                    qwu.PrepareRandomBasis(wires=mappedWire)

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 5.3. If this is not the last layer, no additional measurements are needed, the current quantum state is carried to the next layer    
            elif not layer == self.numQaoaLayers - 1 and wireCutPartIndex == len(self.wireCutPartitions) - 1:
                # No measurements required, quantum state is being passed on the next layer, no action required.
                pass

            # 5.4. If this is the last layer, measure all qubits and add to classical register, exit function with break
            elif self.backend == QuantumBackEnd.PENNYLANE: 
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    classicBitRegister.update({wire: qml.measure(wires=mappedWire, reset=True)})

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)
                
            # 5.5. If this is the last layer but a different backend is selected, use the classicComputer object.
            else:
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    self.classicalComputer.AppendMeasure(wire=mappedWire, reset=True)
 
        return classicBitRegister
    
    def DepolarizationQuantumCircuitReversed(self, gamma: list[float], beta: list[float], layer: int) -> list[int]:
        classicBitRegister: dict = {}
        
        for wireCutPartIndex, wireCutPart in enumerate(self.reversedWireCutPartitions):
            # 0. Get the specified wire cut partition information
            wireCutPartAll = wireCutPart["all"] # Contains all nodes
            wireCutPartContribution = wireCutPart["contribution"] # Contains all nodes which measurement contributes to the final bitstring
            wireCutPartMeasurePrepare = wireCutPart["measure-prepare"] # Contains all nodes which measurement does not contribute to the final bitstring

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 1. Put all the wires (qubits) in the Hadamard basis
            [qml.Hadamard(wires=wire) for wire in range(1, self.numWires+1)]
            
            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 2. Apply cost circuit
            allMin: int = min(wireCutPart["all"]); allMax: int = max(wireCutPart["all"])

            for edge in reversed(list(self.graph.graph.edges)):
                if edge[0] >= allMin and edge[1] <= allMax:
                    # Apply the cost circuit
                    qml.IsingZZ(gamma[layer] * 2, wires=self.MapEdgeReverse(edge, wireCutPartIndex))

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 3. Appy mixer circuit
            for mappedWire in self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex):
                # Apply the RX gate to the qubit
                qml.RX(beta[layer] * 2, wires=mappedWire)

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 4. Apply the depolarization channel on the measure prepare qubits
            for mappedWire in self.MapNodesReverse(wireCutPartMeasurePrepare, partitionIndex=wireCutPartIndex):
                # 4.1. Destroy encoded information on the qubit using a measurement
                if self.backend == QuantumBackEnd.PENNYLANE:
                    qml.measure(wires=mappedWire, reset=True)
                else:
                    self.classicalComputer.Measure(wire=mappedWire, reset=True)
                
                # 4.2. Prepare the qubit in random basis state e.g. (|0>, |1>, |+>, |+i>) for the next part
                qwu.PrepareRandomBasis(wires=mappedWire)

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 5. If this is not the last layer, apply a depolarization channel on the contribution wires
            if not layer == self.numQaoaLayers - 1 and not wireCutPartIndex == len(self.reversedWireCutPartitions) - 1:
                # Apply the depolarization channel on the contribution wires
                for mappedWire in self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex):
                    # 5.1. Destroy encoded information on the qubit using a measurement
                    if self.backend == QuantumBackEnd.PENNYLANE:
                        qml.measure(wires=mappedWire, reset=True)
                    else:
                        self.classicalComputer.Measure(wire=mappedWire, reset=True)
                    
                    # 5.2. Prepare the qubit in random basis state e.g. (|0>, |1>, |+>, |+i>) for the next part
                    qwu.PrepareRandomBasis(wires=mappedWire)

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 5.3. If this is not the last layer, no additional measurements are needed, the current quantum state is carried to the next layer    
            elif not layer == self.numQaoaLayers - 1 and wireCutPartIndex == len(self.reversedWireCutPartitions) - 1:
                # No measurements required, quantum state is being passed on the next layer, no action required.
                pass

            # 5.4. If this is the last layer, measure all qubits and add to classical register.
            elif self.backend == QuantumBackEnd.PENNYLANE: 
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    classicBitRegister.update({wire: qml.measure(wires=mappedWire, reset=True)})
                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)
                
            # 5.5. If this is the last layer but a different backend is selected, use the classicComputer object.
            else:
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    self.classicalComputer.AppendMeasure(wire=mappedWire, reset=True)

        return classicBitRegister


    def RandomCliffordQuantumCircuit(self, gamma: list[float], beta: list[float], layer: int) -> dict:
            nodeInHadamardBasisCounter: int = 0
            classicBitRegister: dict = {}
            
            for wireCutPartIndex, wireCutPart in enumerate(self.wireCutPartitions):
                # 0. Get the speficied wire cut partition information
                wireCutPartAll = wireCutPart["all"] # Contains all nodes
                wireCutPartContribution = wireCutPart["contribution"] # Contains all nodes which measurement contributes to the final bitstring
                wireCutPartMeasurePrepare = wireCutPart["measure-prepare"] # Contains all nodes which measurement does not contribute to the final bitstring

                # Classical register for the measure and prepare channels
                measurePrepareBitRegister: list = []

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

                # 1. Apply the Hadamard gates to the new qubits specific wirecutpart
                # If this is the first partition and not the first layer, no Hadamard gates have to be added
                if wireCutPartIndex == 0 and layer != 0:
                    for wire in wireCutPartAll:
                        if wire > nodeInHadamardBasisCounter:
                            nodeInHadamardBasisCounter += 1
                else:
                    for wire in wireCutPartAll:
                        if wire > nodeInHadamardBasisCounter:
                            qml.Hadamard(wires=self.MapNode(wire, wireCutPartIndex))
                            nodeInHadamardBasisCounter += 1

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

                # 2. Apply cost circuit
                allMin: int = min(wireCutPart["all"]); allMax: int = max(wireCutPart["all"])

                for edge in self.graph.graph.edges:
                    if edge[0] >= allMin and edge[1] <= allMax:
                        # Apply the cost circuit
                        qml.IsingZZ(gamma[layer] * 2, wires=self.MapEdge(edge, wireCutPartIndex))

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

                # 3. Appy mixer circuit 
                for mappedWire in self.MapNodes(wireCutPartContribution, wireCutPartIndex):
                    # Apply the RX gate to the qubit
                    qml.RX(beta[layer] * 2, wires=mappedWire)

                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

                # 4. If this is the last partition and last layer, measure all qubits and add to classical register, exit function with break
                if wireCutPartIndex == len(self.wireCutPartitions) - 1 and layer == self.numQaoaLayers - 1:
                    for wire, mappedWire in zip(wireCutPartContribution, self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                        if self.backend == QuantumBackEnd.PENNYLANE:
                            classicBitRegister.update({wire: qml.measure(wires=mappedWire, reset=False)})
                        else:
                            self.classicalComputer.AppendMeasure(wire=mappedWire, reset=False)   
                    break
                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

                # 5. Apply adjointed Random Clifford circuit for the measure prepare wires
                measurePrepareRandomCliffordOps: list = qwu.RandomCliffordCircuit(wires=self.MapNodes(wireCutPartMeasurePrepare, partitionIndex=wireCutPartIndex), depth=1)
                qwu.ApplyCircuit(measurePrepareRandomCliffordOps, adjoint=True)
                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

                # 6. Apply adjointed Random Clifford circuit for the contribution wires if this is not the last layer.
                if not layer == self.numQaoaLayers - 1 and not wireCutPartIndex == len(self.wireCutPartitions) - 1:
                    contributeRandomCliffordOps: list = qwu.RandomCliffordCircuit(wires=self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex), depth=1)
                    qwu.ApplyCircuit(contributeRandomCliffordOps, adjoint=True)

                    # 6.1. Measure the contribution wires and add to the InitializeMeasurement objects in classiccomputer object
                    initMeasurements: list = []
                    mappedNodes: list = self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex)
                    for mappedWire in mappedNodes:
                        initMeasurements.append(qml.measure(wires=mappedWire, reset=True))
                    self.classicalComputer.StoreMeasurementAndClifford(measurements=initMeasurements, cliffCirc=contributeRandomCliffordOps, layer=layer+1, wires=mappedNodes)       
                    qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)
                    
                # 6.2. If this is not the last layer, no additional measurements are needed, the current quantum state is carried to the next layer    
                elif not layer == self.numQaoaLayers and wireCutPartIndex == len(self.wireCutPartitions) - 1:
                    # No measurements required, quantum state is being passed on the next layer, no action required.
                    pass

                # 6.3 If this is the last layer, measure all qubits and add to classical register, exit function with break
                elif self.backend == QuantumBackEnd.PENNYLANE: 
                    for wire, mappedWire in zip(wireCutPartContribution, self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                        classicBitRegister.update({wire: qml.measure(wires=mappedWire, reset=True)})

                # 6.4 If this is the last layer but a different backend is selected, use the classicComputer object.
                else:
                    for wire, mappedWire in zip(wireCutPartContribution, self.MapNodes(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                        self.classicalComputer.AppendMeasure(wire=mappedWire, reset=True)
                
                # 7. Measure prepare wires and add to measure prepare bitstring register
                counter: int = 0
                for mappedWire in self.MapNodes(wireCutPartMeasurePrepare, partitionIndex=wireCutPartIndex):
                    if self.backend == QuantumBackEnd.PENNYLANE:
                        measurePrepareBitRegister.append(qml.measure(wires=mappedWire, reset=True))
                    else:
                        self.classicalComputer.MeasurePrepare(measureWire=mappedWire, prepareWire=counter, reset=True)  
                        counter += 1

                # 8. Prepare wires based on previous measurements
                if self.backend == QuantumBackEnd.PENNYLANE:
                    wireCounter: int = 1
                    for measurement in measurePrepareBitRegister:
                        qml.cond(measurement, lambda: qml.PauliX(wires=wireCounter))()
                        wireCounter += 1

                # 9. Apply the clifford circuit again but not adjointed
                shiftedOperations = qwu.WireStartFrom(operations=measurePrepareRandomCliffordOps, startFromWire=1)
                qwu.ApplyCircuit(shiftedOperations, adjoint=False)

                # 10. Initliaze qubits with measurements from previous layer and apply the clifford circuits but not adjointed 
                for initMeasurement in self.classicalComputer.initMeasurements:
                    if initMeasurement.layer == layer:
                        # 10.1. Initialize the qubits with the measurements from the previous layer
                        for wire, measurement in zip(initMeasurement.wires, initMeasurement.measurements):
                            qml.cond(measurement, lambda: qml.PauliX(wires=wire))()

                        # 10.2. Apply the same clifford circuit as in the previous layer not adjointed
                        qwu.ApplyCircuit(initMeasurement.cliffCirc, adjoint=False)

                        # 10.3. Remove the init measurement from the list
                        self.classicalComputer.initMeasurements.remove(initMeasurement)

            return classicBitRegister
    
    def RandomCliffordQuantumCircuitReversed(self, gamma: list[float], beta: list[float], layer: int) -> dict:
        nodeInHadamardBasisCounter: int = self.reversedWireCutPartitions[0]["all"][0] + 1
        classicBitRegister: dict = {}
        
        for wireCutPartIndex, wireCutPart in enumerate(self.reversedWireCutPartitions):
            # 0. Get the speficied wire cut partition information
            wireCutPartAll = wireCutPart["all"] # Contains all nodes
            wireCutPartContribution = wireCutPart["contribution"] # Contains all nodes which measurement contributes to the final bitstring
            wireCutPartMeasurePrepare = wireCutPart["measure-prepare"] # Contains all nodes which measurement does not contribute to the final bitstring

            # Classical register for the measure and prepare channels
            measurePrepareBitRegister: list = []

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 1. Apply the Hadamard gates to the new qubits specific wirecutpart
            # If this is the first partition and not the first layer, no Hadamard gates have to be added
            if wireCutPartIndex == 0 and layer != 0:
                for wire in wireCutPartAll:
                    if wire > nodeInHadamardBasisCounter:
                        nodeInHadamardBasisCounter += 1
            else:
                for wire in wireCutPartAll:
                    if wire > nodeInHadamardBasisCounter:
                        qml.Hadamard(wires=self.MapNodeReverse(wire, wireCutPartIndex))
                        nodeInHadamardBasisCounter += 1

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 2. Apply cost circuit
            allMin: int = min(wireCutPart["all"]); allMax: int = max(wireCutPart["all"])

            for edge in reversed(list(self.graph.graph.edges)):
                if edge[0] >= allMin and edge[1] <= allMax:
                    # Apply the cost circuit
                    qml.IsingZZ(gamma[layer] * 2, wires=self.MapEdgeReverse(edge, wireCutPartIndex))

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 3. Appy mixer circuit 
            for mappedWire in self.MapNodesReverse(wireCutPartContribution, wireCutPartIndex):
                # Apply the RX gate to the qubit
                qml.RX(beta[layer] * 2, wires=mappedWire)

            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 4. If this is the last partition and last layer, measure all qubits and add to classical register, exit function with break
            if wireCutPartIndex == len(self.wireCutPartitions) - 1 and layer == self.numQaoaLayers - 1:
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    if self.backend == QuantumBackEnd.PENNYLANE:
                        classicBitRegister.update({wire: qml.measure(wires=mappedWire, reset=False)})
                    else:
                        self.classicalComputer.AppendMeasure(wire=mappedWire, reset=False)   
                break
            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 5. Apply adjointed Random Clifford circuit for the measure prepare wires
            measurePrepareRandomCliffordOps: list = qwu.RandomCliffordCircuit(wires=self.MapNodesReverse(wireCutPartMeasurePrepare, partitionIndex=wireCutPartIndex), depth=1)
            qwu.ApplyCircuit(measurePrepareRandomCliffordOps, adjoint=True)
            qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)

            # 6. Apply adjointed Random Clifford circuit for the contribution wires if this is not the last layer.
            if not layer == self.numQaoaLayers - 1 and not wireCutPartIndex == len(self.wireCutPartitions) - 1:
                contributeRandomCliffordOps: list = qwu.RandomCliffordCircuit(wires=self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex), depth=1)
                qwu.ApplyCircuit(contributeRandomCliffordOps, adjoint=True)

                # 6.1. Measure the contribution wires and add to the InitializeMeasurement objects in classiccomputer object
                initMeasurements: list = []
                mappedNodes: list = self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex)
                for mappedWire in mappedNodes:
                    initMeasurements.append(qml.measure(wires=mappedWire, reset=True))
                self.classicalComputer.StoreMeasurementAndClifford(measurements=initMeasurements, cliffCirc=contributeRandomCliffordOps, layer=layer+1, wires=mappedNodes)       
                qml.Barrier(wires=list(range(1, self.numWires+1)), only_visual=True)
                
            # 6.2. If this is not the last layer, no additional measurements are needed, the current quantum state is carried to the next layer    
            elif not layer == self.numQaoaLayers and wireCutPartIndex == len(self.wireCutPartitions) - 1:
                # No measurements required, quantum state is being passed on the next layer, no action required.
                pass

            # 6.3 If this is the last layer, measure all qubits and add to classical register, exit function with break
            elif self.backend == QuantumBackEnd.PENNYLANE: 
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    classicBitRegister.update({wire: qml.measure(wires=mappedWire, reset=True)})

            # 6.4 If this is the last layer but a different backend is selected, use the classicComputer object.
            else:
                for wire, mappedWire in zip(wireCutPartContribution, self.MapNodesReverse(wireCutPartContribution, partitionIndex=wireCutPartIndex)):
                    self.classicalComputer.AppendMeasure(wire=mappedWire, reset=True)
            
            # 7. Measure prepare wires and add to measure prepare bitstring register
            counter: int = 0
            for mappedWire in self.MapNodesReverse(wireCutPartMeasurePrepare, partitionIndex=wireCutPartIndex):
                if self.backend == QuantumBackEnd.PENNYLANE:
                    measurePrepareBitRegister.append(qml.measure(wires=mappedWire, reset=True))
                else:
                    self.classicalComputer.MeasurePrepare(measureWire=mappedWire, prepareWire=counter, reset=True)  
                    counter += 1

            # 8. Prepare wires based on previous measurements
            if self.backend == QuantumBackEnd.PENNYLANE:
                wireCounter: int = self.numWires
                for measurement in measurePrepareBitRegister:
                    qml.cond(measurement, lambda: qml.PauliX(wires=wireCounter))()
                    wireCounter -= 1

            # 9. Apply the clifford circuit again but not adjointed
            shiftedOperations = qwu.WireStartFromReversed(operations=measurePrepareRandomCliffordOps, startFromWire=self.numWires)
            qwu.ApplyCircuit(shiftedOperations, adjoint=False)

            # 10. Initliaze qubits with measurements from previous layer and apply the clifford circuits but not adjointed 
            for initMeasurement in self.classicalComputer.initMeasurements:
                if initMeasurement.layer == layer:
                    # 10.1. Initialize the qubits with the measurements from the previous layer
                    for wire, measurement in zip(initMeasurement.wires, initMeasurement.measurements):
                        qml.cond(measurement, lambda: qml.PauliX(wires=wire))()

                    # 10.2. Apply the same clifford circuit as in the previous layer not adjointed
                    qwu.ApplyCircuit(initMeasurement.cliffCirc, adjoint=False)

                    # 10.3. Remove the init measurement from the list
                    self.classicalComputer.initMeasurements.remove(initMeasurement)

        return classicBitRegister
    
    def ToMeasurements(self, measurements: dict) -> list:
        """
        Transfroms the measurement dictionary from the random clifford and depolarization channel to a sorted list from smallest wire left to largest wire right.

        Args:
            measurements (dict): The measurements to be transforme; returned from random clifford and depolarization quantum circuit functions.
        
        Returns:
            list: The transformed measurements ordered starting with the smallest wire.
        """
        register: list = []
        sortedMeasurements = sorted(measurements.items())

        for index in sortedMeasurements:
            register.append(index[1])

        return register

    
    def __PennylaneRandomCliffordQuantumCircuit__(self, gamma: list[float], beta: list[float]):
        classicBitRegister: dict = None
        reverseCostCircuit: bool = False

        for layer in range(self.numQaoaLayers):
            
            if reverseCostCircuit:
                classicBitRegister = self.ToMeasurements(self.RandomCliffordQuantumCircuitReversed(gamma, beta, layer))
                reverseCostCircuit = False
            else:
                classicBitRegister = self.ToMeasurements(self.RandomCliffordQuantumCircuit(gamma, beta, layer))
                reverseCostCircuit = True
        
        # Return counts for minimization on index [0] and probs for visualization on index [1]
        if not self.backend == QuantumBackEnd.PENNYLANE:
            return None
        return qml.counts(op=classicBitRegister)
    
    def __PennylaneDepolarizationQuantumCircuit__(self, gamma: list[float], beta: list[float]):
        classicBitRegister: dict = None
        reverseCostCircuit: bool = False

        for layer in range(self.numQaoaLayers):
            
            if reverseCostCircuit:
                classicBitRegister = self.ToMeasurements(self.DepolarizationQuantumCircuitReversed(gamma, beta, layer))
                reverseCostCircuit = False
            else:
                classicBitRegister = self.ToMeasurements(self.DepolarizationQuantumCircuit(gamma, beta, layer))
                reverseCostCircuit = True
        
        # Return counts for minimization on index [0] and probs for visualization on index [1]
        if not self.backend == QuantumBackEnd.PENNYLANE:
            return None
        return qml.counts(op=classicBitRegister)

    def __str__(self) -> str:
        """
        Returns a string representation of the circuit.
        """
        outputStr: str = f"WireCutCircuit: \n"
        outputStr += str(self.graph)
        outputStr += str(self.config)
        return outputStr

    def Visualise(self, backend: QuantumBackEnd = QuantumBackEnd.PENNYLANE, channel: QuantumChannel = QuantumChannel.RANDOM_CLIFFORD) -> None:
        """
        Visualize and print the quantum circuit.
        """
        self.backend = backend
        parameters = [0.5] * self.numQaoaLayers
        if backend == QuantumBackEnd.PENNYLANE:

            rndCliffDevice = qml.device("lightning.qubit", wires=self.wires, shots=1000)
            depolarDevice = qml.device("lightning.qubit", wires=self.wires, shots=1000)

            self.rndCliffQnode = qml.QNode(self.__PennylaneRandomCliffordQuantumCircuit__, rndCliffDevice, mcm_method="one-shot")
            self.depolarQnode = qml.QNode(self.__PennylaneDepolarizationQuantumCircuit__, depolarDevice, mcm_method="one-shot")

            if channel == QuantumChannel.RANDOM_CLIFFORD:
                print(qml.draw(qnode=self.rndCliffQnode, max_length=250)(parameters, parameters))
            elif channel == QuantumChannel.DEPOLARIZATION:
                print(qml.draw(qnode=self.depolarQnode, max_length=250)(parameters, parameters))
            else:
                raise ValueError(f"Channel not supported; received: {channel}")	
        else:
            qasm3 = self.ToQasm3(parameters, parameters)
            if channel == QuantumChannel.RANDOM_CLIFFORD:
                circ: QuantumCircuit = parse(qasm3[0])
            elif channel == QuantumChannel.DEPOLARIZATION:
                circ: QuantumCircuit = parse(qasm3[1])
            else:
                raise ValueError(f"Channel not supported; received: {channel}")
            
            print(circ.draw("text", fold=150))
        
    def __fetchNoiseModel__(self, backend: QuantumBackEnd) -> NoiseModel:
        service = QiskitRuntimeService()
        print("Fetching noise model from IMBQ backend; this might take a while...")
        match backend:
            case QuantumBackEnd.QISKIT_AER_IBM_BRISBANE:
                backend = service.backend(name="ibm_brisbane", instance="ibm-q/open/main")
            case QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
                backend = service.backend(name="ibm_sherbrooke", instance="ibm-q/open/main")
            case _:
                raise Exception(f"Backend does not require noise model or is not supported; received: {backend.value}")
        print(f"Noise model: {backend.value} fetched successfully.")
        return NoiseModel.from_backend(backend)
    
    def __setupQiskitAer__(self, backend: QuantumBackEnd) -> None:
        """"
        Creates the AerSimulator object for the Qiskit backend with, if specified, required noise model.
        """
        match backend:
            case QuantumBackEnd.QISKIT_AER:
                self.aerSim = AerSimulator(executor={'num_threads': 4}, method='automatic')
            case QuantumBackEnd.QISKIT_AER_IBM_BRISBANE | QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE: 
                model = NoiseManager.ReadNoiseModel(backend)
                if model is not None:
                    # If the noise model file exists, use it
                    self.aerSim = AerSimulator(noise_model=model, executor={'num_threads': 4}, method='automatic')
                else:
                    # If the noise model file does not exist, fetch it from IBM Quantum Platform
                    print("Fetching noise model from IMBQ backend; this might take a while...")
                    model = NoiseManager.FetchNoiseModel(backend)
                    # Save the noise model to a binary file to save time in the future
                    NoiseManager.WriteNoiseModel(model, backend)
                    self.aerSim = AerSimulator(noise_model=model, executor={'num_threads': 4}, method='automatic')
            case _:
                # If another backend is provided, no action required.
                pass

    def __runQiskitAerCircuit__(self, channel: QuantumChannel, gamma: list[float], beta: list[float]) -> dict:
        # Qiskit Aer simulator and if requested noise models are already setup.
        qasm3 = self.ToQasm3(gamma, beta)
        
        if channel == QuantumChannel.RANDOM_CLIFFORD:
            circ: QuantumCircuit = parse(qasm3[0]) 
            job = self.aerSim.run(circ, shots=self.numShotsRndCliffCircuit)
        else:
            circ: QuantumCircuit = parse(qasm3[1])
            job = self.aerSim.run(circ, shots=self.numShotsDepolarCircuit)
            
        # remove the ancillary register from the output counts
        jobResultCounts = job.result().get_counts(circ)
        countsDict = {}
        for stateDecimal in range(2**self.numOriginalQubits):
            # Loop through all possible states
            # Convert the decimal state to binary
            stateBinary = qu.ToBinary(self.numOriginalQubits, stateDecimal)

            # Loop through the job result counts keys
            counts: int = 0
            for key in jobResultCounts.keys():
                cleanedKey = key[2:] # Remove the ancillary register (remove first two chars 0 or 1 and " ".)
                
                if cleanedKey == stateBinary:
                    counts += jobResultCounts[key]

            countsDict.update({stateBinary: counts})

        return countsDict

    def __calculateCircuitShots(self, contribution: tuple[int, int]) -> None:
        """
        Distributes the shot budget among both channel and calculates the number of circuit variants.
        To mitigate shot noise, the constant MIN_SHOTS_PER_CIRCUIT is used to ensure that each circuit has a minimum of 100 shots.
        """
        # MIN_SHOTS_PER_CIRCUIT: int = self.shotsBudget // 10 # e.g. for a budget of 2000 each circuit has at minimum 200 shots.

        MIN_SHOTS_PER_CIRCUIT: int = 200 # e.g. for a budget of 2000 each circuit has at minimum 200 shots.


        self.numRndCliffCircuits: int = contribution[0] // MIN_SHOTS_PER_CIRCUIT
        remainder: int = contribution[0] % MIN_SHOTS_PER_CIRCUIT
        self.numShotsRndCliffCircuit: int = remainder // self.numRndCliffCircuits + MIN_SHOTS_PER_CIRCUIT

        self.numDepolarCircuits: int = contribution[1] // MIN_SHOTS_PER_CIRCUIT
        remainder: int = contribution[1] % MIN_SHOTS_PER_CIRCUIT
        self.numShotsDepolarCircuit: int = remainder // self.numDepolarCircuits + MIN_SHOTS_PER_CIRCUIT

        # Used for debugging:
        # raise Exception(f"Number of circuits for random clifford channel: {self.numRndCliffCircuits} with {self.numShotsRndCliffCircuit} shots per circuit.\n"
        #                 f"Number of circuits for depolarization channel: {self.numDepolarCircuits} with {self.numShotsDepolarCircuit} shots per circuit.\n"
        #                 f"Total shots budget: {self.shotsBudget}.\n")

    def __RunQuantumCircuit__(self, parameters) -> float | tuple[list, dict]:
        """
        Runs the quantum circuit with the specified parameters and returns the cost or probabilities & counts.
        Args:
            parameters (list): The parameters for the quantum circuit (gamma and beta).
        Returns:
            Returns float or tuple[list, dict]: The cost of the quantum circuit or the (list) probabilities[0] and (dict) counts[1].
        """
        gamma = parameters[:self.numQaoaLayers]
        beta = parameters[self.numQaoaLayers:]

        
        channelContribution = self.__computeChannelContribution__()
        self.__calculateCircuitShots(channelContribution)

        # Setup Pennylane backend
        if self.backend == QuantumBackEnd.PENNYLANE:
            rndCliffDevice = qml.device("lightning.qubit", wires=self.wires, shots=self.numShotsRndCliffCircuit)
            depolarDevice = qml.device("lightning.qubit", wires=self.wires, shots=self.numShotsDepolarCircuit)

            self.rndCliffQnode = qml.QNode(self.__PennylaneRandomCliffordQuantumCircuit__, rndCliffDevice, mcm_method="one-shot")
            self.depolarQnode = qml.QNode(self.__PennylaneDepolarizationQuantumCircuit__, depolarDevice, mcm_method="one-shot")

        finalResultsCounts: dict = {}; sumCountsChn0: dict = {}; sumCountsChn1: dict = {}

        for state in range(2**self.numOriginalQubits):
            finalResultsCounts.update({qu.ToBinary(self.numOriginalQubits, state): 0})
            sumCountsChn0.update({qu.ToBinary(self.numOriginalQubits, state): 0})
            sumCountsChn1.update({qu.ToBinary(self.numOriginalQubits, state): 0})


        # 1. Run the random Clifford channel
        for i in range(self.numRndCliffCircuits):
            countChn0: dict = {}
            if self.backend == QuantumBackEnd.PENNYLANE:
                countChn0 = self.rndCliffQnode(gamma, beta)
            elif self.backend == QuantumBackEnd.QISKIT_AER or self.backend == QuantumBackEnd.QISKIT_AER_IBM_BRISBANE or self.backend == QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
                countChn0 = self.__runQiskitAerCircuit__(QuantumChannel.RANDOM_CLIFFORD, gamma, beta)
            else:
                raise NotImplementedError("IBMQ not supported yet")
            
            for state in sumCountsChn0.keys():
                if state in countChn0:
                    sumCountsChn0[state] += countChn0[state]

        # 2. Run the depolarization channel
        for i in range(self.numDepolarCircuits):
            countsChn1: dict = {}
            if self.backend == QuantumBackEnd.PENNYLANE:
                countsChn1 = self.depolarQnode(gamma, beta)
            elif self.backend == QuantumBackEnd.QISKIT_AER or self.backend == QuantumBackEnd.QISKIT_AER_IBM_BRISBANE or self.backend == QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
                countsChn1 = self.__runQiskitAerCircuit__(QuantumChannel.DEPOLARIZATION, gamma, beta)
            else:
                raise NotImplementedError("IBMQ not supported yet")

            for state in sumCountsChn1.keys():
                if state in countsChn1:
                    sumCountsChn1[state] += countsChn1[state]

        # Used for debugging.
        # print(f"Channel 0: {sumCountsChn0}")
        # print(f"Channel 1: {sumCountsChn1}")

        # 4. Apply equation for contribution of the channels
        for state in finalResultsCounts:
            result = (self.d + 1) * sumCountsChn0[state] - self.d * sumCountsChn1[state]
            if result < 0:
                result = 0
            finalResultsCounts[state] = result

        # 5. Normalize the results
        probabilities = qu.ConvertCountsToProbabilities(finalResultsCounts)

        finalResultsCounts = {key: 0 for key in finalResultsCounts} # reset counts dictionary

        counter: int = 0
        for key, value in finalResultsCounts.items():
            finalResultsCounts[key] = probabilities[counter] * self.shotsBudget
            counter += 1	

        # TODO, found bug, when qiskit aer selected and solving graph b all counts are mirrored: 0101 -> 1010
        # temporary fix, remove this when bug is fixed
        if self.graph.name == "B" and self.backend != QuantumBackEnd.PENNYLANE:
            finalResultsCounts = qu.MirrorCounts(finalResultsCounts)

        # 6. Return either the cost or the probabilities[0] and counts[1]
        if self.hasOptimizedParameters:
            return probabilities, finalResultsCounts
        else:
            cost: float = QaoaUtils.CostFunction(self.graph.graph, finalResultsCounts)
            if cost > self.optimalAvgCost:
                self.optimalAvgCost = cost
                self.optimalParameters = parameters
            self.costHistory.append(cost)
            return -cost # Negative due minimization, SciPy does not offer maximize functionality for COBYLA.

    def __ComputeOptimizedParameters__(self) -> dict:
        """
        Optimize the parameters for the QAOA quantum circuit.

        Returns:
            dict: A dictionary containing the optimized parameters and the cost history: {gamma, beta, costHistory}.
        """
        self.costHistory.clear()

        # Initial guess for gamma and beta parameters
        gammaGuess = [0.5] * self.numQaoaLayers
        betaGuess = gammaGuess
        self.hasOptimizedParameters = False

        if self.config.config["classicalOptimizationAlgorithm"] == "COBYLA":
            self.config.config["classicalOptimizationProvider"] = "SciPy"
            MinimizeScipy(self.__RunQuantumCircuit__, gammaGuess + betaGuess, method="COBYLA", options={"maxiter": self.config.config["maxClassicalOptimizationIterations"]})
        elif self.config.config["classicalOptimizationAlgorithm"] == "SPSA":
            self.config.config["classicalOptimizationProvider"] = "PiPy"

            # TODO, parameters are configured for gamma and beta * 2. 
            # Later this changed to gamma * 2 and beta * 2, so the parameters are not optimal anymore.

            match self.graph.name:
                case "A":
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.2, lr_decay=0.602, px_decay=0.166 , px=0.3)    
                    # MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.18, lr_decay=0.602, px_decay=0.166 , px=0.2)   
                case "B":
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.25, lr_decay=0.602, px_decay=0.166 , px=0.4)    
                case "C":
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.25, lr_decay=0.602, px_decay=0.166 , px=0.45)    
                case _:
                    # Default case, use default paramters
                    print("Unrecognized graph, using default graph settings")
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.18, lr_decay=0.602, px_decay=0.166 , px=0.2)   
        else:
            raise ValueError(f"Classical optimization algorithm not supported; received: {self.config.config['classicalOptimizationAlgorithm']}")

        self.hasOptimizedParameters = True
        return {"gamma": self.optimalParameters[:self.numQaoaLayers], "beta" : self.optimalParameters[self.numQaoaLayers:], "costHistory" : self.costHistory}


    def Run(self, backend: QuantumBackEnd = QuantumBackEnd.PENNYLANE) -> ExperimentResult:
        self.backend = backend
        self.config.config["quantumCircuitBackend"] = backend.value

        self.__setupQiskitAer__(backend)

        self.hasOptimizedParameters = False

        # Only used for converting to qasm if Pennylane is not selected
        self.classicalComputer = ClassicalComputer()

        optimizedParams = self.__ComputeOptimizedParameters__()
        
        self.config.config["optimalAvgCost"] = self.optimalAvgCost
        self.config.config["gamma"] = optimizedParams.get("gamma")
        self.config.config["beta"] = optimizedParams.get("beta")
        self.config.config["numClassicalOptimizationIterations"] = len(optimizedParams.get("costHistory"))
        
        combinedParams = [*optimizedParams.get("gamma"), *optimizedParams.get("beta")]
        resultDict = self.__RunQuantumCircuit__(combinedParams)

        # Used optimal parameteres to run the circuit
        self.hasOptimizedParameters = False
        return ExperimentResult(self.graph, self.config, data={"Gamma" : optimizedParams.get("gamma"), "Beta" : optimizedParams.get("beta"), "CostHistory": optimizedParams.get("costHistory"), "Probabilities" : resultDict[0], "Counts" : resultDict[1]})
    
    def __computeCutQubits__(self) -> int:
        numCutQbits: int = 0

        if self.numQaoaLayers > 1:
            # The cut wires on contribution wires should also be counted
            for index in range(len(self.wireCutPartitions) - 1):
                partition = self.wireCutPartitions[index]
                numCutQbits += len(partition["contribution"])

        # All contribution wires are being cut except the last partition when their are mutiple layers

        for partition in self.wireCutPartitions:
            numCutQbits += len(partition["measure-prepare"])
        return numCutQbits

    def __computeChannelContribution__(self) -> tuple[float, float]:
        self.d = 2**self.numCutQubits
        chnProb0 = (self.d + 1) / (2 * self.d + 1) # Random Clifford channel
        chnProb1 = self.d / (2 * self.d + 1) # Depolarization channel

        return round(chnProb0 * self.shotsBudget), round(chnProb1 * self.shotsBudget)
    
    def __upgradeQasmHeader__(self, qasmArr: list[str]) -> str:
        """
        Converts the qasm 2.0 header to qasm 3.0. with: version, quantum and classical registers.
        """
        qasmArr[0] = qasmArr[0].replace("2.0;", "3.0;") # From "OPENQASM 2.0;" to "OPENQASM 3.0;"
        qasmArr[1] = qasmArr[1].replace("qelib1.inc", "stdgates.inc") # From "include qelib1.inc;" to "include stdgates.inc;"

        # Define Qubit register
        qasmArr[2] = f"qreg q[{self.numWires}];\n" # From example: "qreg q[5];" to "qubit q[5];"
        # Define classical bit register
        qasmArr[3] = f"creg c[{len(self.graph.graph.nodes)}];\n" # From example: "creg c[5];" to "bit c[5];"

        # Add Ancillary register with a single bit
        qasmArr.insert(4, "creg a[1];\n") # Add ancillary classical register of a single bit.

        return qasmArr


    def ToQasm3(self, gamma: list[float], beta: list[float]) -> tuple[str, str]:
        """
        Converts the circuit to QASM 3.0 format, including mid circuit measurements.
        """
        self.backend = QuantumBackEnd.QISKIT_AER

        # Construct the classical computer object which holds the measurements to the classical register
        self.classicalComputer = ClassicalComputer()

        with qml.tape.QuantumTape() as rndCliffTape:
            self.__PennylaneRandomCliffordQuantumCircuit__(gamma, beta)
        
        rndCliffArr = rndCliffTape.to_openqasm(measure_all=False).splitlines(keepends=True) # Returns OPENQASM 2.0 format of the Pennylane circuit.

        # update syntax for qasm 3.0
        rndCliffArr = self.__upgradeQasmHeader__(rndCliffArr)

        measurementCounter: int = 0
        for qasmLineIndex, qasmLine in enumerate(rndCliffArr):
            if qasmLine.startswith("id q["):
                rndCliffArr[qasmLineIndex] = self.classicalComputer.measurements[measurementCounter].ToQasm()
                measurementCounter += 1

        # Reconstruct new classical computer object which holds the measurements to the classical register
        self.classicalComputer = ClassicalComputer()

        with qml.tape.QuantumTape() as depolarTape:
            self.__PennylaneDepolarizationQuantumCircuit__(gamma, beta)
        
        depolarArr = depolarTape.to_openqasm(measure_all=False).splitlines(keepends=True)

        depolarArr = self.__upgradeQasmHeader__(depolarArr)

        measurementCounter = 0
        for qasmLineIndex, qasmLine in enumerate(depolarArr):
            if qasmLine.startswith("id q["):
                depolarArr[qasmLineIndex] = self.classicalComputer.measurements[measurementCounter].ToQasm()
                measurementCounter += 1

        return "".join(rndCliffArr), "".join(depolarArr)
        