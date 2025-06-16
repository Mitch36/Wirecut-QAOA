import networkx as nx
from quantum_utils import QuantumUtils as qu

class QaoaUtils:
    @staticmethod
    def ComputeEdgesCrossed(graph: nx.Graph, bitStr: str) -> int:
        group0: list[int] = []; group1: list[int] = []
        for qubit, char in enumerate(bitStr):
            if char == "0":
                group0.append(qubit+1) # Account for zero indexing
            else:
                group1.append(qubit+1) # Account for zero indexing

        edgeCrossedCounter: int = 0
        for edge in graph.edges():
            if edge[0] in group0 and edge[1] in group0:
                # Do nothing, edge is not crossed
                pass
            elif edge[0] in group1 and edge[1] in group1:
                # Do nothing, edge is not crossed
                pass
            else:
                edgeCrossedCounter += 1
        
        return edgeCrossedCounter

    @staticmethod
    def CostFunction(graph: nx.Graph, results: dict[str, int]) -> float:
        # results are in format key: bitstring, value: count
        totalCost: float = 0
        totalResults: int = 0
        for key, value in results.items():
            edgesCrossed = QaoaUtils.ComputeEdgesCrossed(graph, key)
            totalCost += edgesCrossed * value
            totalResults += value

        return totalCost / totalResults     
    
    @staticmethod
    def MergeInverses(results: dict) -> dict:
        """
        Merges the inverses results of the QAOA MAXCUT problem. Example: 010 and 101 are inverses, so they are summed into 010.
        """

        if not isinstance(results, dict):
            raise ValueError("Results must be a dictionary.")

        if len(results) % 2 != 0:
            raise ValueError(f"Results dictionary must have an even number of elements since it must contain binary strings; received: {len(results)} elements.")

        mergedResults: dict = {}
        counter: int = 0
        for key, value in results.items():
            if counter == len(results)/2:
                break
            mergedResults.update({key: value+results[qu.FlipBits(key)]})
            counter += 1

        return mergedResults

    @staticmethod
    def GetBestSolutions(solutions: dict) -> list[str]:
        """
        Creates a subset of the solutions where the most edges are being cut.

        Returns:
            list[str]: List of bistrings representing the best solutions
        """

        if not isinstance(solutions, dict):
            raise ValueError("Solutions must be a dictionary.")

        maxCut: int = max(solutions.values())
        bestSolutions: list[str] = []
        for key, value in solutions.items():
            if value == maxCut:
                bestSolutions.append(key)
        return bestSolutions
