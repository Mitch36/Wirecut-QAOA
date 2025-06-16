# Internal libraries
from graph import Graph
from experiment_configuration import ExperimentConfiguration as ExpConf
from qaoa_utils import QaoaUtils
from experiment_result import ExperimentResult
from quantum_backend import QuantumBackEnd
from quantum_utils import QuantumUtils as qu
from noise_manager import NoiseManager

# External libraries
from scipy.optimize import minimize as MinimizeScipy # Used in classical optimization loop
from spsa import minimize as MinimizeSPSA # Used in classical optimization loop
import pennylane as qml
from copy import deepcopy # Used for copying the configuration

# Qiskit Libraries
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

class Circuit:
    def __init__(self, graph: Graph, config: ExpConf):
        self.graph: Graph = graph
        self.config: ExpConf = deepcopy(config)

        # Update required qubits parameter in the configuration
        self.config.config["numQubits"] = len(graph.graph.nodes)
        self.config.config["numOriginalQubits"] = len(graph.graph.nodes)
        # No wire cutting is applied, update config var to 0
        self.config.config["numCuts"] = 0

        self.numLayers: int = config.Get("numQaoaLayers")
        self.wires: list[int] = list(range(1, len(graph.graph.nodes) + 1))
        self.numWires: int = len(self.wires)
        self.shotsBudget: int = config.Get("shotsBudgetPerQuantumCircuit")

        # Classical optimization bookeeping
        self.optimalAvgCost: float = 0.0
        self.optimalParameters: list = []
        self.costHistory: list[float] = []
        self.hasOptimizedParameters: bool = False

        # Setup Pennylane backend
        device = qml.device("lightning.qubit", wires=self.wires, shots=self.shotsBudget)
        self.qnode = qml.QNode(self.__PennylaneQuantumCircuit__, device)

    def Run(self, backend: QuantumBackEnd = QuantumBackEnd.PENNYLANE) -> ExperimentResult:
        self.backend = backend
        self.config.config["quantumCircuitBackend"] = backend.value
        self.hasOptimizedParameters = False

        self.__setupQiskitAer__(backend)
        optimizedParams = self.__ComputeOptimizedParameters__()

        self.config.config["optimalAvgCost"] = self.optimalAvgCost
        self.config.config["gamma"] = optimizedParams.get("gamma")
        self.config.config["beta"] = optimizedParams.get("beta")
        self.config.config["numClassicalOptimizationIterations"] = len(optimizedParams.get("costHistory"))

        combinedParams = [*optimizedParams.get("gamma"), *optimizedParams.get("beta")]

        resultDict = self.__RunQuantumCircuit__(combinedParams)

        return ExperimentResult(self.graph, self.config, data={"Gamma" : optimizedParams.get("gamma"), "Beta" : optimizedParams.get("beta"), "CostHistory": optimizedParams.get("costHistory"), "Probabilities" : resultDict[0], "Counts" : resultDict[1]})
    
    def __setupQiskitAer__(self, backend: QuantumBackEnd) -> None:
        """"
        Creates the AerSimulator object for the Qiskit backend with, if specified, required noise model.
        """
        match backend:
            case QuantumBackEnd.QISKIT_AER:
                self.aerSim = AerSimulator()
            case QuantumBackEnd.QISKIT_AER_IBM_BRISBANE | QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE: 
                model = NoiseManager.ReadNoiseModel(backend)
                if model is not None:
                    # If the noise model file exists, use it
                    self.aerSim = AerSimulator(noise_model=model)
                else:
                    # If the noise model file does not exist, fetch it from IBM Quantum Platform
                    model = NoiseManager.FetchNoiseModel(backend)
                    # Save the noise model to a binary file to save time in the future
                    NoiseManager.WriteNoiseModel(model, backend)
                    self.aerSim = AerSimulator(noise_model=model)
            case _:
                # If another backend is provided, no action required.
                pass

    def __PennylaneQuantumCircuit__(self, gamma: list[float], beta: list[float]):
        # Set all qubits to the Hamdard state: |+>
        [qml.Hadamard(wires=w) for w in self.wires]

        reverse: bool = False

        if len(gamma) != len(beta):
            raise ValueError(f"gamma and beta must be of the same length; received: gamma = {len(gamma)}, beta = {len(beta)}")

        for layer in range(self.numLayers):
            # Do cost circuit, apply gamma parameter
            if reverse:
                for index, edge in enumerate(reversed(list(self.graph.graph.edges()))):
                    qml.IsingZZ(gamma[layer] * 2, wires=edge)
                reverse: bool = False

            else:
                for edge in self.graph.graph.edges:
                    qml.IsingZZ(gamma[layer], wires=edge)
                reverse: bool = True

            qml.Barrier(wires=self.wires)
            # Do mixer circuit, apply beta parameter
            [qml.RX(beta[layer] * 2, wires=w) for w in self.wires]
            qml.Barrier(wires=self.wires)

        # Return counts for minimization on index [0] and probs for visualization on index [1]
        return qml.counts(wires=self.wires), qml.probs(wires=self.wires)

    def __toQiskitCircuit__(self, gamma: list[float] = None, beta: list[float] = None) -> QuantumCircuit:
        """"
        Converts Pennylane circuit to Qiskit circuit through OPENQASM 2.0
        """
        if gamma is None or beta is None:
            gamma = [0.5] * self.numLayers
            beta = gamma

        with qml.tape.QuantumTape() as tape:
            self.__PennylaneQuantumCircuit__(gamma, beta)
        qasm = tape.to_openqasm(measure_all=False)
        return QuantumCircuit.from_qasm_str(qasm)

    def __RunQuantumCircuit__(self, parameters) -> float | tuple[list, dict]:
        """
        Runs the quantum circuit with the given parameters and returns the cost or probabilities & counts.

        Args:
            parameters (list): The parameters for the quantum circuit (gamma and beta).

        Returns:
            float | tuple[list, dict]: The cost of the quantum circuit or the probabilities and counts.

        """

        # Make distinction between gamma and beta parameters
        gamma = parameters[:self.numLayers]
        beta = parameters[self.numLayers:]

        match self.backend:
            case QuantumBackEnd.PENNYLANE:
                result = self.qnode(gamma, beta)
                counts = result[0]
                probs = result[1]
            case QuantumBackEnd.QISKIT_AER | QuantumBackEnd.QISKIT_AER_IBM_BRISBANE | QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
                qiskitCircuit = self.__toQiskitCircuit__(gamma, beta)
                job = self.aerSim.run(qiskitCircuit, shots=self.shotsBudget)
                counts = job.result()
                counts = counts.get_counts(qiskitCircuit)
            case QuantumBackEnd.IBMQ_BRISBANE | QuantumBackEnd.IBMQ_SHERBROOKE:
                raise NotImplementedError("IBMQ not supported")
            case _:
                raise Exception(f"Backend not supported; received: {self.backend.value}")

        if self.hasOptimizedParameters:
            if not self.backend == QuantumBackEnd.PENNYLANE:
                cleanCounts = qu.CleanCounts(counts)

                # TODO, found bug, when qiskit aer selected and solving graph b all counts are mirrored: 0101 -> 1010
                # temporary fix, remove this when bug is fixed
                if self.graph.name == "B" and not self.backend == QuantumBackEnd.PENNYLANE:
                    cleanCounts = qu.MirrorCounts(cleanCounts)

                return qu.ConvertCountsToProbabilities(cleanCounts), cleanCounts
            else:
                return probs, qu.CleanCounts(counts) # Pennylane already returns the probs in clean format but not the counts 
        else:
            cleanedCounts = qu.CleanCounts(counts)
            if self.graph.name == "B" and not self.backend == QuantumBackEnd.PENNYLANE:
                cleanedCounts = qu.MirrorCounts(cleanedCounts)
            cost: float = QaoaUtils.CostFunction(self.graph.graph, cleanedCounts)
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
        gammaGuess = [0.5] * self.numLayers
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
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.25, lr_decay=0.602, px_decay=0.166 , px=0.3)                
                case "B":
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.25, lr_decay=0.602, px_decay=0.166 , px=0.3)    
                case "C":
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.3, lr_decay=0.602, px_decay=0.105 , px=0.3)    
                case _:
                    # Default case, use default paramters
                    MinimizeSPSA(self.__RunQuantumCircuit__, gammaGuess + betaGuess, iterations=self.config.config["maxClassicalOptimizationIterations"], lr=0.15, lr_decay=0.602, px_decay=0.166 , px=0.3)    
        else:
            raise ValueError(f"Classical optimization algorithm not supported; received: {self.config.config['classicalOptimizationAlgorithm']}")

        self.hasOptimizedParameters = True
        return {"gamma": self.optimalParameters[:self.numLayers], "beta" : self.optimalParameters[self.numLayers:], "costHistory" : self.costHistory}
    
    def Visualise(self, backend: QuantumBackEnd):
        """
        Visualize and print the quantum circuit.
        """
        parameters = [0.5] * self.numLayers
        if backend == QuantumBackEnd.PENNYLANE:
                print(qml.draw(qnode=self.qnode, max_length=250)(parameters, parameters))
        else:
            qiskitCircuit = self.__toQiskitCircuit__(parameters, parameters)
            print(qiskitCircuit.draw("text", fold=150))

    def __str__(self) -> str:
        """
        Returns a string representation of the circuit.
        """
        outputStr: str = f"Circuit: \n"
        outputStr += str(self.graph)
        outputStr += str(self.config)
        return outputStr