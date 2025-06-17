from experiment_configuration import ExperimentConfiguration as ExpConf
from graph import Graph, Rank

from quantum_utils import QaoaUtils

from matplotlib import pyplot as plt

class ExperimentResult:
    def __init__(self, graph: Graph, config: ExpConf, data: dict):
        self.graph: Graph = graph
        self.config: ExpConf = config
        self.data: dict = data

    def GetProbs(self) -> list:
        """
        Returns the probabilities of the experiment result.
        """
        return self.data.get("Probabilities", None)
    
    def GetCounts(self) -> dict:
        """
        Returns the final counts of the experiment result.
        """
        return self.data.get("Counts", None)
    
    def GetShotsBudget(self) -> int:
        """
        Returns the shots budget for the experiment result.
        """
        return self.config.Get("shotsBudgetPerQuantumCircuit")
    
    def RankSolution(self) -> Rank:
        """
        Returns Rank of the given solution, Ranks are defined by Rank enum in Graph.py.
        """
        mergedCounts = QaoaUtils.MergeInverses(self.GetCounts())
        return self.graph.RankSolution(max(mergedCounts, key=mergedCounts.get))
    
    def GetCostHistory(self) -> list:
        """
        Returns the cost history of the experiment result.
        """
        # raise Exception(self.data.keys())
        return self.data.get("CostHistory", [0])
    
    def GetGamma(self) -> list[float]:
        """
        Returns the gamma values used in the experiment result.
        """
        return self.data.get("Gamma", [0])
    
    def GetBeta(self) -> list[float]:
        """
        Returns the beta values used in the experiment result.
        """
        return self.data.get("Beta", [0])
    
    def GetOptimizedParameters(self) -> list[float]:
        """
        Returns the optimized parameters used in the experiment result.
        """
        return self.GetGamma() + self.GetBeta()
    
    def GetExperimentName(self) -> str:
        """
        Returns the name of the experiment.
        """
        return self.config.Get("experimentName")	

    def GetApproximateRatio(self) -> float:
        """
        Returns the approximate ratio of the experiment result.

        Returns:
            (float) Value found by the algorithm divided by the optimal value.
        """

        counts = self.GetCounts()
        shots = self.GetShotsBudget()
        sum: int = 0
        for key, value in counts.items():
            sum += value * self.graph.solutions.get(key, 0)
        return sum / (self.graph.maxCut * shots)
    
    def PlotCostHistory(self) -> None:
        plt.plot(self.GetCostHistory())
        plt.title("Cost history of: " + self.GetExperimentName())
        plt.ylabel("Cost")
        plt.xlabel("Iterations")
        plt.grid(True)
        plt.show()

    def __str__(self) -> str:
        outputStr: str = f"ExperimentResult: \n"
        outputStr += str(self.graph)
        outputStr += str(self.config)
        outputStr += str(self.data)
        return outputStr

        