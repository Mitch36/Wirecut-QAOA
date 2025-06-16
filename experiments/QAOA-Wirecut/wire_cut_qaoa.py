import networkx as nx
from experiment_configuration import ExperimentConfiguration as ExpConf
from graph import Graph
from quantum_backend import QuantumBackEnd
from experiment_result import ExperimentResult
from quantum_utils import QuantumUtils as qu

from circuit import Circuit
from wire_cut_circuit import WireCutCircuit
from copy import deepcopy

from IPython.display import display
import ipywidgets as widgets

class WireCutQAOA:
    def __init__(self, graph: Graph, config: ExpConf):
        self.graph: Graph = graph
        self.config: ExpConf = deepcopy(config)

    def RunUncut(self, backend: QuantumBackEnd) -> list[ExperimentResult]:
        """
        Runs the uncut QAOA circuit for the given graph and configuration.
        Args:
            backend (QuantumBackEnd): The quantum backend to use for the simulation.
        Returns:
            list[ExperimentResult]: A list of ExperimentResult objects containing the results of the uncut QAOA circuit of each sample.
        """

        sampleSize: int = self.config.Get("numSamples")

        progressBar = widgets.IntProgress(min=0, max=sampleSize, description="Running Uncut QAOA Circuit", bar_style="info")
        display(progressBar)

        originalExperimentName: str = self.config.Get("experimentName")

        experimentResults: list[ExperimentResult] = []

        for sample in range(sampleSize):
            # Update the configuration for each sample
            self.config.config["experimentName"] = f"{originalExperimentName} {sample}"
            
            # Run the circuit
            result: ExperimentResult = Circuit(self.graph, self.config).Run(backend)
            experimentResults.append(result)
            progressBar.value += 1
        return experimentResults
    
    def RunWireCut(self, backend: QuantumBackEnd) -> list[ExperimentResult]:
        """
        Runs the Wirecut QAOA circuit for the given graph and configuration.
        Args:
            backend (QuantumBackEnd): The quantum backend to use for the simulation.
        Returns:
            list[ExperimentResult]: A list of ExperimentResult objects containing the results of the uncut QAOA circuit of each sample.
        """

        sampleSize: int = self.config.Get("numSamples")

        progressBar = widgets.IntProgress(min=0, max=sampleSize, description="Running Wircut QAOA Circuit", bar_style="info")
        display(progressBar)

        originalExperimentName: str = self.config.Get("experimentName")

        experimentResults: list[ExperimentResult] = []

        for sample in range(sampleSize):
            # Update the configuration for each sample
            self.config.config["experimentName"] = f"{originalExperimentName} {sample}"
            
            # Run the circuit
            result: ExperimentResult = WireCutCircuit(self.graph, self.config).Run(backend)
            experimentResults.append(result)
            progressBar.value += 1
        return experimentResults