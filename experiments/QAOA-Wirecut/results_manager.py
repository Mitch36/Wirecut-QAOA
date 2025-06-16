from experiment_result import ExperimentResult
import pickle
import os 

from graph import Rank

class Benchmark:
    def __init__(self, name: str, data: list[ExperimentResult]):
        
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        if len(name) < 1:
            raise ValueError("Name must be a non-empty string.")
        if len(name) > 100:
            raise ValueError("Name must be a string of length less than 100 characters.")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of ExperimentResult objects.")
        if len(data) == 0:
            raise ValueError("Data list cannot be empty.")
        if not all(isinstance(result, ExperimentResult) for result in data):
            raise ValueError("All elements in data must be of type ExperimentResult.")
        
        self.name: str = name
        self.data: list[ExperimentResult] = data
        self.approximationRatios: list[float] = [result.GetApproximateRatio() for result in data]
        self.approximationRatio: float = self.__computeApproximationRatio__()
        self.successProbalities: dict = self.__computeSuccesProbabilities__()

    def GetApproximationRatio(self) -> float:
        return self.approximationRatio
    
    def GetApproximationRatios(self) -> list[float]:
        return self.approximationRatios

    def GetSuccessProbabilities(self) -> dict:
        return self.successProbalities

    def __computeApproximationRatio__(self) -> float:
        """
        Computes the approximation ratio of the benchmark.
        """
        if len(self.data) == 0:
            raise ValueError("Data list cannot be empty.")
        
        return sum(self.approximationRatios) / len(self.data)
    
    def __computeSuccesProbabilities__(self) -> dict:
        """
        Computes the success probability of the benchmark.

        Returns:
            dict: A dictionary containing the success probabilities for each rank, ranks are defined in Graph.py in the Rank Enum.
        """
        bestCounter: int = 0; secondBestCounter: int = 0; thirdBestCounter: int = 0; wrongCounter: int = 0
        
        for result in self.data:
            if not isinstance(result, ExperimentResult):
                raise ValueError("All elements in data must be of type ExperimentResult.")

            rank: Rank = result.RankSolution()
            match rank:
                case Rank.BEST:
                    bestCounter += 1
                case Rank.SECOND_BEST:
                    secondBestCounter += 1
                case Rank.THIRD_BEST:
                    thirdBestCounter += 1
                case Rank.WRONG:
                    wrongCounter += 1
                case _:
                    raise ValueError(f"Unknown rank: {rank}")

        # Convert counts to probabilities
        totalCount: int = len(self.data)
        return {
            "Best": bestCounter / totalCount,
            "Second Best": secondBestCounter / totalCount,
            "Third Best": thirdBestCounter / totalCount,
            "Wrong": wrongCounter / totalCount
            }
    
    def __str__(self) -> str:
        outputString = f"Benchmark: {self.name}\n"
        outputString += f"\tApproximation Ratio: {self.approximationRatio}\n"
        outputString += f"\tSuccess Probabilities: {self.successProbalities}\n"
        return outputString
    
    def ToString(self) -> str:
        return self.__str__()


class ResultsManager:
    def __init__(self):
        self.benchmarks: list[Benchmark] = []

    @staticmethod
    def FromBinaryFile(filename: str = "bin/resultsmanager.pkl"):
        """
        Reads the results from a binary file.
        """
        manager: ResultsManager = ResultsManager()
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                object: ResultsManager = pickle.load(file)
                manager.benchmarks = object.benchmarks
                return manager
        else:
            raise Exception(f"File does not exist: {filename}")

    def AddResults(self, name: str, results: list[ExperimentResult]) -> None:
        """
        Adds a result to the results manager.
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        if len(name) < 1:
            raise ValueError("Name must be a non-empty string.")
        if len(name) > 100:
            raise ValueError("Name must be a string of length less than 100 characters.")
        if not isinstance(results, list):
            raise ValueError("Results must be a list of ExperimentResult objects.")
        if len(results) == 0:
            raise ValueError("Results list cannot be empty.")
        if not all(isinstance(result, ExperimentResult) for result in results):
            raise ValueError("All elements in results must be of type ExperimentResult.")
        
        # Check if the benchmark already exists
        for index, benchmark in enumerate(self.benchmarks):
            if benchmark.name == name:
                # Overwrite the already existing benchmark with the new results
                self.benchmarks[index] = Benchmark(name, results)
                # raise NotImplementedError(f"Benchmark with name {name} already exists. Use AddResults to add more results; Needs to be implemented.")

        self.benchmarks.append(Benchmark(name, results))

    def Get(self, name: str) -> Benchmark:
        """
        Gets a benchmark by name.
        """
        for benchmark in self.benchmarks:
            if benchmark.name == name:
                return benchmark
        raise ValueError(f"Benchmark with name {name} not found.")
    
    def GetAll(self) -> list[Benchmark]:
        """
        Gets all benchmarks.
        """
        return self.benchmarks
    
    def GetAllForGraph(self, graphName: str) -> list[Benchmark]:
        graphBenchmarks: list[Benchmark] = []

        for benchmark in self.benchmarks:
            if graphName in benchmark.name:
                graphBenchmarks.append(benchmark)

        return graphBenchmarks

    def __str__(self) -> str:
        outputString = "ResultsManager: "
        for benchmark in self.benchmarks:
            outputString += f"\n\t{benchmark.name}: {len(benchmark.data)} results"
        return outputString

    def WriteToBinaryFile(self, filename: str = "bin/resultsmanager.pkl") -> str:
        """
        Writes the results to a binary file.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return filename	

