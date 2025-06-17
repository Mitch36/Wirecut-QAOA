# Class which holds the benchmark data
# Can be exported by pickle to binary file or CSV to text file.

from abc import ABC, abstractmethod
import pickle
from quantum_utils.src.quantum_utils import QuantumUtils as qu

class DataEntry(ABC):
    """
    Abstract base class for data entries in the benchmark.
    """
    @abstractmethod
    def __init__(self, data: list):
        pass
    @abstractmethod
    def __str__(self):
        pass
    @abstractmethod
    def ToCSVString(self) -> str:
        pass

class Benchmark():
    def __init__(self):
        self.data: list[DataEntry] = []
        self.headers: list[str] = []
        self.AddHeader("Index")

    def AddHeader(self, header: str):
        """
        Adds a header to the benchmark.
        """
        self.headers.append(header)

    def SetHeaders(self, headers: list[str]):
        """
        Sets the headers of the benchmark.
        """
        self.headers = headers
    
    def GetHeaders(self) -> list[str]:
        """
        Returns the headers of the benchmark.
        """
        result: str = ""
        for index, header in enumerate(self.headers):
            if index != len(self.headers) - 1:
                result += str(header) + ";"
            else:
                result += str(header) + "\n"
        return result
       
    def AddDataEntry(self, dataEntry: DataEntry):
        """
        Adds a data entry to the benchmark.
        """
        index = len(self.data)
        dataEntry.data.insert(0, index)
        self.data.append(dataEntry)

    def CompareGraphs(self, fieldX: str, fieldY: str) -> bool:
        # 1. Check if the headers match the fields
        try: 
            indexX = self.headers.index(fieldX)
            indexY = self.headers.index(fieldY)
        except ValueError:
            raise ValueError("Invalid field names; fields must be in the headers")
        
        # 2. Fetch data from both headers
        dataX = []; dataY = []
        for data in self.data:
            dataX.append(data.GetAt(indexX))
            dataX.append(data.GetAt(indexY))

        # 3. Check if the data is equal
        if len(dataX) != len(dataY):
            raise ValueError("Invalid data; data lengths do not match, data entries must be equal in the number of values")

        # 4. Plot graph
    def GetHeaderIndex(self, headerToSearch: str) -> int:
        """
        Returns the index of the header of the benchmark.

        Args:
            headerToSearch (str): The header to search for.
        Returns:
            int: The index of the header. if not found returns -1
        """
        index = -1
        for index, header in enumerate(self.headers):
            if header == headerToSearch:
                return index
        
        return -1

    def GetWhere(self, headerToSearchFrom: str, keyValue, headerToObtainValuesFrom: str) -> list:
        """
        Gets from the (headerToObtainValuesFrom) parameter header values whenever the dataentry matches the keyValue parameter in the (headerToSearchFrom) parameter.
        """
        headerToSearchFromIndex = self.GetHeaderIndex(headerToSearchFrom)
        headerToObtainValuesFromIndex = self.GetHeaderIndex(headerToObtainValuesFrom)

        if headerToSearchFromIndex == -1 or headerToObtainValuesFromIndex == -1:
            raise ValueError("Invalid header; header does not exist")
        
        results = []
        for data in self.data:
            if data.data[headerToSearchFromIndex] == keyValue:
                results.append(data.data[headerToObtainValuesFromIndex])

        return results 

    
    def ExportToCSV(self, fileName: str = None):
        """
        Exports the benchmark data to a CSV file.
        """
        if fileName[-4:] != ".csv":
            fileName += ".csv"
        with open(fileName, "w") as file:
            file.write(self.GetHeaders())
            for data in self.data:
                file.write(data.ToCSVString())
    
    def ExportToBinary(self, fileName: str = None):
        """
        Exports the benchmark data to a binary file.
        """
        if len(fileName) == 0:
            fileName = "benchmark_data.pkl"

        if fileName[-4:] != ".pkl":
            fileName += ".pkl"

        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def ImportFromBinary(fileName: str) -> "Benchmark":
        """
        Imports the benchmark data from a binary file.
        """
        if fileName[-4:] != ".pkl":
            fileName += ".pkl"

        with open(fileName, "rb") as file:
            return pickle.load(file)


class BenchmarkResult(DataEntry):
    def __init__(self, data: list):
        self.data: list = data

    def __str__(self):
        return f"BenchmarkDataEntry: {self.data}"
    
    def GetAt(self, index: int) -> any:
        """
        Returns the data entry.
        """
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range.")
        return self.data[index]	
    
    def ToCSVString(self) -> str:
        """
        Converts the data entry to a CSV string.
        """
        result: str = ""
        for index, value in enumerate(self.data):
            result += str(value)
            if index < len(self.data) - 1:
                result += ";"
            else:
                result += "\n"
        return result

    