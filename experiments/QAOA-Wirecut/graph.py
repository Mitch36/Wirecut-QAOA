import networkx as nx
from matplotlib import pyplot as plt # For visualization of the graph

from qaoa_utils import QaoaUtils
from quantum_utils import QuantumUtils as qu
       

from enum import Enum
class Rank(Enum):
    """
    Enum to represent the rank of the provided solution, used by Graph.RankSolution() function
    """
    BEST = 1, "Best"
    SECOND_BEST = 2, "Second Best"	
    THIRD_BEST = 3, "Third Best"	
    WRONG = 4, "Wrong"

    def __str__(self):
        return self.value[1]

class Graph:
    def __init__(self, rows: list[int], name: str = "Graph"):

        if len(rows) < 3:
            raise ValueError("Graph must have at least 3 rows to apply wire cutting to it.")

        self.name: str = name
        self.rows = rows
        self.graph: nx.Graph = nx.Graph()
        counter: int = 1
        customPositions: dict = {}
        widestLayer: int = max(rows)

        color: bool = True # Used for colour distinction between layers

        layers: list[list[int]] = [] # Contains list where each node is in, used for calculating the edges

        rows.reverse() # Rows are reversed since later in this algorithm we need to start from the bottom

        verticalPositions: list[int] = []
        numRows = len(rows)
        for i in range(numRows):
            # Calculate the layer number
            verticalPositions.append(numRows - i - 1)
            # Results in layers = [7, 6, ... 1, 0]

        # we now have divided each layer, we can loop through it and assign the nodes
        for verticalPos in verticalPositions:
            layers.append([])
            for node in range(rows[verticalPos]):
                if rows[verticalPos] < widestLayer:
                    indent = (widestLayer - rows[verticalPos]) * 0.5
                    customPositions.update({counter: (node + indent, verticalPos)})
                else:
                    customPositions.update({counter: (node, verticalPos)})
                # Make colour distinction between layers
                if color:
                    self.graph.add_node(counter, color='tab:blue')
                else:
                    self.graph.add_node(counter, color='tab:red')
                layers[-1].append(counter)
                counter += 1

            if color:
                color = False
            else:
                color = True

        self.positions = customPositions

        # Added edges to the neighbouring nodes

        isEven: bool = True
        for currentLayerIndex, layer in enumerate(layers):
            if isEven:
                # First connect to neighbouring nodes in the same layer
                
                # create edges to the right neighbouring nodes in the same layer
                for layerIndex, node in enumerate(layer):
                    # check if the node is not the last in the layer
                    if not layerIndex == len(layer) - 1:
                        self.graph.add_edge(node, node + 1)
                isEven = False
            else:
                # create edges to the neighbouring nodes in the previous layer and next layer if applicable
                # loop through current layer
                for node in layer:
                    # Loop through the previous layer
                    for prevLayerIndex, prevLayerNode in enumerate(layers[currentLayerIndex-1]):
                        self.graph.add_edge(node, prevLayerNode)
                    
                    # First check if their is a next layer
                    if not currentLayerIndex == len(layers) - 1:
                        # There is a next layer, Loop through the next layer
                        for nextLayerIndex, nextLayerNode in enumerate(layers[currentLayerIndex+1]):
                            self.graph.add_edge(node, nextLayerNode)
                isEven = True 

        self.solutions: dict = self.__computeSolutions__() # Dict containing each cut for each state
        self.__RankSolutions__() # Computes and add the solutions to each solution list

    def Visualise(self, title: str = "Graph") -> None:
        
        nodeColors = [self.graph.nodes[node]['color'] for node in self.graph.nodes()]

        nx.draw_networkx_nodes(self.graph, self.positions, node_color=nodeColors, node_size=500)
        nx.draw_networkx_edges(self.graph, self.positions, edge_color='gray',width=2 )
        nx.draw_networkx_labels(self.graph, self.positions, font_size=12, font_weight='bold', font_color='whitesmoke')

        plt.title(title)
        plt.axis('on')
        plt.show()

        print(f"The most optimal solutions for {self.name} are: {self.bestSolutions}")
        print(f"The maximum cut is: {self.maxCut}")

    def __computeSolutions__(self) -> dict:
        """
        Computes the best solutions for the graph MAXCUT problem

        Returns:
            dict: Dictionary with the best solutions
        """
        solutions: dict = {}

        for state in range(2**len(self.graph.nodes)):
            # Loop through all possible states
            binaryState = qu.ToBinary(len(self.graph.nodes), state)
            cuts = QaoaUtils.ComputeEdgesCrossed(self.graph, binaryState)
            solutions.update({binaryState: cuts})
        
        return solutions

    def __RankSolutions__(self) -> tuple[int, int, int]:
        """
        Computes the best, second best, third best solutions for the graph MAXCUT problem.
        Also sets the maxCut, secondMaxCut and thirdMaxCut variables.

        Returns:
            tuple: Triple int tuple containing the best, second best and third best solutions
        """
        if self.solutions is None:
            raise ValueError("Solutions have not been computed yet, please run __computeSolutions__() first.")
        
        self.maxCut: int = max(self.solutions.values())
        self.secondMaxCut: int = 0; self.thirdMaxCut: int = 0
        self.bestSolutions: list[str] = []; self.secondBestSolutions: list[str] = []; self.thirdBestSolutions: list[str] = []

        # Loop through the solutions and find the best, second best and third best solutions
        for cut in range(self.maxCut-1, 0, -1):

            if self.secondMaxCut != 0 and self.thirdMaxCut != 0:
                break

            for key, value in self.solutions.items():
                if value == cut and self.secondMaxCut == 0:
                    self.secondMaxCut = cut
                    break
                elif value == cut and self.thirdMaxCut == 0:
                    self.thirdMaxCut = cut
                    break    

        # Second best maxcut solution and third best maxcut solution are found, now add the binary strings to the corrects lists

        for binaryStr, cuts in self.solutions.items():
            if cuts == self.maxCut:
                self.bestSolutions.append(binaryStr)
            elif cuts == self.secondMaxCut:
                self.secondBestSolutions.append(binaryStr)
            elif cuts == self.thirdMaxCut:
                self.thirdBestSolutions.append(binaryStr)

        return self.maxCut, self.secondMaxCut, self.thirdMaxCut
    
    def RankSolution(self, bitStr: str) -> Rank:
        """
        Computes the rank of the provided bistring, ranks are defined by the Rank enum. Ranked from best to worst:
        1. Best
        2. Second best
        3. Third best
        4. Wrong

        Args:
            bitStr (str): The bitstring to rank.
        Returns:
            Rank : The rank of the provided bitstring.
        """
    
        if self.solutions is None:
            raise ValueError("Solutions have not been computed yet, please run __computeSolutions__() first.")
        if self.maxCut is None:
            raise ValueError("Maxcut has not been computed yet, please run __RankSolutions__ first.")
        
        if bitStr in self.bestSolutions:
            return Rank.BEST
        elif bitStr in self.secondBestSolutions:
            return Rank.SECOND_BEST
        elif bitStr in self.thirdBestSolutions:
            return Rank.THIRD_BEST
        else:
            return Rank.WRONG

    def GetMaxCut(self) -> int:
        """
        Returns the bitstring which has the most edges crossed (maxcut).
        """
        if self.maxCut == 0:
            raise ValueError("Maxcut has not been computed yet, please run __computeBestSolutions__() first.")
        return self.maxCut

    def __str__(self):
        outputStr = f"Graph {self.name}: \n"
        for node in self.graph.nodes():
            outputStr += f"Node {node}: {self.graph.nodes[node]}\n"
        for edge in self.graph.edges():
            outputStr += f"Edge {edge}: {self.graph.edges[edge]}\n"
        return outputStr
