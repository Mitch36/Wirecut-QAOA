# from abc import staticmethod

import matplotlib.pyplot as plt
from pennylane.numpy import random
from pennylane import numpy as np

class QuantumUtils:

    @staticmethod
    def SetSeed(seed: int):
        """
        Sets the seed for random number generation

        Args:
            seed (int): Seed value
        """
        random.seed(seed)

    @staticmethod
    def ToBinary(numQubits: int, decimal: int) -> str:
        """
        Converts a decimal number to a binary string with a fixed number of qubits

        Args:
            numQubits (int): Number of qubits
            decimal (int): Decimal number to convert to binary

        Returns:
            str: Binary string

        Raises:
            Exception: Invalid parameter: numQubits; must be greater than 0
            Exception: Invalid parameter: decimal; must be positive number
        """
        if numQubits < 1:
            raise Exception("Invalid parameter: numQubits; must be greater than 0")
        if decimal < 0:
            raise Exception("Invalid parameter: decimal; must be positive number")

        binaryString = str(bin(decimal)[2:])
        numQubits = numQubits - len(binaryString)
        for i in range(numQubits):
            binaryString = "0" + binaryString
        return binaryString

    @staticmethod
    def ToDecimal(binaryStr: str) -> float:
        """
        Converst a binary string to a decimal number

        Args:
            binaryStr (str): Binary string to convert to decimal

        Returns:
            int: Decimal number

        Raises:
            ValueError: Invalid character in binaryStr: {binaryStr} must be either 0 or 1
        """
        # First clean up string by removing '|' and '>'
        index = binaryStr.find("|")
        if index != -1:
            binaryStr = binaryStr[:index] + binaryStr[index + 1 :]
        index = binaryStr.find(">")
        if index != -1:
            binaryStr = binaryStr[:index] + binaryStr[index + 1 :]
        # String is now cleaned of '|' and '>'
        decimal = 0
        power = 0
        for digit in reversed(binaryStr):
            if digit == "0":
                pass
            elif digit == "1":
                decimal += 2**power
            else:
                raise ValueError("Invalid character in binaryStr: " + binaryStr + " must be either 0 or 1")
            power += 1
        return decimal
    
    @staticmethod
    def FlipBits(binaryStr: str) -> str:
        """
        Inverses the bits in a binary string

        Args:
            binaryStr (str): Binary string to inverse

        Returns:
            str: Inversed binary string

        Raises:
            Exception: Invalid parameter: binaryStr; empty string
            Exception: Invalid character in state: {binaryStr} must be either 0 or 1
        """
        binaryArr = list(binaryStr)
        for i, char in enumerate(binaryArr):
            if char == "0":
                binaryArr[i] = "1"
            elif char == "1":
                binaryArr[i] = "0"
            else:
                raise Exception("Invalid character in state: " + binaryStr + " must be either 0 or 1")
        return "".join(binaryArr)

    @staticmethod
    def FlipBit(binaryStr: str, index: int) -> str:
        """
        Flips a bit in a binary string at a given index"

        Args:
            binaryStr (str): Binary string to flip a bit in
            index (int): Index of the bit to flip, negative index is allowed to count from the end of the string

        Returns:
            str: Binary string with the bit flipped at the given index

        Raises:
            Exception: Invalid parameter: binaryStr; empty string
            Exception: Invalid parameter: index; out of bounds
            Exception: Invalid character in state: {binaryStr} must be either 0 or 1

        """
        strLength = len(binaryStr)
        if strLength == 0:
            raise Exception("Invalid parameter: binaryStr; empty string")
        if index < strLength * -1 or index >= strLength:
            raise Exception("Invalid parameter: index; out of bounds")

        # Convert binaryStr to char array (list)
        char_list = list(binaryStr)

        if char_list[index] == "0":
            char_list[index] = "1"
        elif char_list[index] == "1":
            char_list[index] = "0"
        else:
            raise Exception("Invalid character in state: " + binaryStr + " must be either 0 or 1")
        return "".join(char_list)

    @staticmethod
    def CreateQuantumStateLabels(numQubits: int) -> list[str]:
        """
        Creates a list, based on the number of qubits, containing all possible quantum states

        Args:
            numQubits (int): Number of qubits

        Returns:
            list[str]: list with all possible quantum states based on the number of qubits parameter
        """

        states = []
        numberOfStates = 2**numQubits

        for q in range(numberOfStates):
            states.append(QuantumUtils.ToBinary(numQubits, q))
        return states
    
    @staticmethod
    def MirrorState(bitStr: str) -> str:
        """
        Mirrors bitstring e.g. 0101 -> 1010
        """
        if not isinstance(bitStr, str):
            raise ValueError("Parameter invalid: bitStr; must be a string")
        
        return "".join(reversed(bitStr))

    @staticmethod
    def MirrorCounts(countsList: dict) -> dict:
        """
        Mirrors the bitstring results e.g. ['01': 5, '10': 3] -> ['01': 3, '10': 5]

        Warning:
            Can only be used with proper cleaned counts, which contain all possible states, use CleanCounts() first.
        """
        
        # Do not loop through entire list, only half due swapping
        counter: int = len(countsList) / 2
        
        for key, value in countsList.items():
            # Mirror the key
            mirroredKey = QuantumUtils.MirrorState(key)

            # Swap the values
            countsList[key] = countsList.get(mirroredKey, 0)
            countsList[mirroredKey] = value
            counter -= 1
            if counter == 0:
                break
        return countsList



    @staticmethod
    def GetNumQubits(numStates: int) -> int:
        """
        Calculates the number of qubits based on the number of states.
        Args:
            numStates (int): Number of states
        Returns:
            int: Number of qubits
        Raises: 
            Exception: Parameter invalid: numStates; invalid length compared to possible qubits
        """
        entries = numStates
        number = 1
        multiplicationCount = 0
        while number < entries:
            number *= 2
            multiplicationCount += 1
            if number == entries:
                break
            elif number > entries:
                raise Exception("Parameter invalid: numStates; invalid length compared to possible qubits")

        return multiplicationCount
    
    @staticmethod
    def Normalize(data: list) -> list:
        """
        Normalizes a list of numbers to sum to 1.
        Args:
            list (list): List of numbers to normalize
        Returns:
            list: Normalized list of numbers
        Raises:
            ValueError: If the input list is empty.
        Note:
            It is not guaranteed that the function returns a list of which the sum is exact 1, however it returns very close to 1.
        """
        if len(data) == 0:
            raise ValueError("Parameter invalid: list; empty list")

        # Accumalte for division by zero exceptions
        total = sum(data)
        if total == 0:
            raise ValueError("Parameter invalid: list; sum of list is 0")

        normalizedData = [x / total for x in data]
          
        return normalizedData
    
    @staticmethod
    def DifferenceSum(contentA: list, contentB: list) -> float:
        """
        Calculates the sum of the absolute differences between two lists of numbers.
        Note: Accepts negative numbers.

        Args:
            contentA (list): First list of numbers.
            contentB (list): Second list of numbers.
        Returns:
            float: The sum of the absolute differences.
        Raises:
            ValueError: If either list is empty or if the lists are of different sizes.

        """
        if len(contentA) == 0:
            raise ValueError("Parameter invalid: contentA, is empty list")
        if len(contentB) == 0:
            raise ValueError("Parameter invalid: contentB, is empty list")
        
        if len(contentA) != len(contentB):
            raise ValueError("Parameter invalid: contentA and contentB, datasets must be equal in size")

        totalDiff = 0
        for valA, valB in zip(contentA, contentB):
            diff = valA - valB
            if diff < 0:
                diff *= -1
            totalDiff += diff
        return totalDiff
    
    @staticmethod
    def HellingerDistance(p: list, q: list) -> float:
        """
        Hellinger distance is a measure of the similarity between two probability distributions.
        It is defined as:
            H(P, Q) = (1 / sqrt(2)) * sqrt( ∑( ( sqrt(p_i) - sqrt(q_i) ))^2 )
        where p_i and q_i are the probabilities of the i-th event in distributions P and Q, respectively.

        Args:
            p (list): The first probability distribution.
            a (list): The second probability distribution.
        Returns:
            float: The Hellinger distance between the two distributions.
        Raises:
            ValueError: If the input distributions are not valid (e.g., empty, different lengths, or not summing to 1).
            ValueError: If the input distributions contain negative values.
        """
        if len(p) == 0 or len(q) == 0:
            raise ValueError("Both probability distributions must be non-empty.")
        if len(p) != len(q):
            raise ValueError("Both probability distributions must have the same length.")
        if 0.9999 < sum(p) > 1.0001 or 0.9999 < sum(q) > 1.0001:
            raise ValueError("Both probability distributions must sum to 1 or very close to it")

        diff  = 0
        for valA, valB in zip(p, q):
            if valA < 0 or valB < 0:
                raise ValueError("Probability values must be non-negative.")
            sqDiff = np.sqrt(valA) - np.sqrt(valB)
            diff += sqDiff ** 2
        return np.sqrt(diff) * (1 / np.sqrt(2))
    
    def CustomPlot(dataLabels: list[str], xAxisLabels: list[str], dataSets: list[list], title: str = "Custom Plot Chart", horizontalLabel: str = "X-axis", verticalLabel: str = "Y-axis"):
        """
        Creates and prints a custom plot of the data

        Args:
            data (list): List of values to plot
            title (str): Title of the graph
            horizontalLabel (str): Label for the x-axis
            verticalLabel (str): Label for the y-axis
        """
        # First convert variable to correct types, if provided list make it into list[list]
        if not isinstance(dataSets[0], list) and not isinstance(dataSets[0], np.ndarray):
            dataSets = [dataSets]
            if len(dataSets[0]) == 0:
                raise ValueError("Parameter invalid: probabilities; empty list")
        # Probabilities is a list of lists, so we need to check if it is empty
        elif any(len(prob) == 0 for prob in dataSets):
            raise ValueError("Parameter invalid: probabilities; some list is empty")

        # dataLabels is a string, make it into list[str]
        if isinstance(dataLabels, str):
            dataLabels = [dataLabels]
        # dataLabels is already a list, so we need to check if it is empty
        elif len(dataLabels) == 0:
            raise ValueError("Parameter invalid: dataLabels; empty list")
        
        # EXPERIMENTAL
        horizontalAxisPos = np.arange(len(xAxisLabels))

        plt.figure(figsize=(8, 4))
        plt.xticks(horizontalAxisPos, xAxisLabels)
        plt.title(title)
        plt.xlabel(horizontalLabel)
        plt.ylabel(verticalLabel)
        plt.grid(True)
        plt.style.use("default")

        match len(dataSets):
            case 1:
                plt.plot(dataSets[0], marker='o', linestyle='-', color='b', label=dataLabels[0], markersize=7)
            case 2:
                plt.plot(dataSets[0], marker='o', linestyle='-', color='b', label=dataLabels[0], markersize=7)
                plt.plot(dataSets[1], marker='s', linestyle='-', color='r', label=dataLabels[1], markersize=7)
            case 3:
                plt.plot(dataSets[0], marker='o', linestyle='-', color='b', label=dataLabels[0], markersize=7)
                plt.plot(dataSets[1], marker='s', linestyle='-', color='r', label=dataLabels[1], markersize=7)
                plt.plot(dataSets[2], marker='^', linestyle='-', color='g', label=dataLabels[2], markersize=7)
            case 4:
                plt.plot(dataSets[0], marker='o', linestyle='-', color='b', label=dataLabels[0], markersize=7)
                plt.plot(dataSets[1], marker='s', linestyle='-', color='r', label=dataLabels[1], markersize=7)
                plt.plot(dataSets[2], marker='^', linestyle='-', color='g', label=dataLabels[2], markersize=7)
                plt.plot(dataSets[3], marker='d', linestyle='-', color='y', label=dataLabels[3], markersize=7)
            case 5:
                plt.plot(dataSets[0], marker='o', linestyle='-', color='b', label=dataLabels[0], markersize=7)
                plt.plot(dataSets[1], marker='s', linestyle='-', color='r', label=dataLabels[1], markersize=7)
                plt.plot(dataSets[2], marker='^', linestyle='-', color='g', label=dataLabels[2], markersize=7)
                plt.plot(dataSets[3], marker='d', linestyle='-', color='y', label=dataLabels[3], markersize=7)
                plt.plot(dataSets[4], marker='v', linestyle='-', color='k', label=dataLabels[4], markersize=7)
            case _:
                raise ValueError("Invalid parameter: dataSets; too many datasets; max is five")

        plt.legend()
        plt.show()
    
    @staticmethod
    def CustomBarChart(dataLabels: str | list[str], xAxisLabels: list[str], dataSets: list | list[list], title: str = "Custom Bar Chart", horizontalLabel: str = "X-axis", verticalLabel: str = "Y-axis"):
        """
        Creates and prints a bar chart of a comparison of at maximum five different datasets

        Args:
            dataLabels (list[str]): List of labels for the data
            xAxisLabels (list[str]): List of labels for the x-axis
            dataSets (list[float]): List of values to plot
            title (str): Title of the graph
            horizontalLabel (str): Label for the x-axis
            verticalLabel (str): Label for the y-axis

        Raises:
            Exception: Invalid parameter: dataLabels; empty list; max is 5
            Exception: Invalid parameter: xAxisLabels; empty list
            Exception: Invalid parameter: values; empty list
        """

        # First convert variable to correct types, if provided list make it into list[list]
        if not isinstance(dataSets[0], list) and not isinstance(dataSets[0], np.ndarray):
            dataSets = [dataSets]
            if len(dataSets[0]) == 0:
                raise ValueError("Parameter invalid: probabilities; empty list")
        # Probabilities is a list of lists, so we need to check if it is empty
        elif any(len(prob) == 0 for prob in dataSets):
            raise ValueError("Parameter invalid: probabilities; some list is empty")

        # dataLabels is a string, make it into list[str]
        if isinstance(dataLabels, str):
            dataLabels = [dataLabels]
        # dataLabels is already a list, so we need to check if it is empty
        elif len(dataLabels) == 0:
            raise ValueError("Parameter invalid: dataLabels; empty list")

        if len(dataLabels) == 0:
            raise Exception("Invalid parameter: dataLabels; empty list")
        if len(xAxisLabels) == 0:
            raise Exception("Invalid parameter: xAxisLabels; empty list")
        if len(dataSets) == 0:
            raise Exception("Invalid parameter: values; empty list")
        
        if (len(dataLabels) * len(xAxisLabels)) == len(dataSets):
            raise ValueError("Invalid parameter: values must contain the duplication of both labels lists")
    
        horizontalAxisPos = np.arange(len(xAxisLabels))

        plt.figure(figsize=(16, 4))
        plt.xticks(horizontalAxisPos, xAxisLabels)
        plt.title(title)
        plt.xlabel(horizontalLabel)
        plt.ylabel(verticalLabel)
        plt.style.use("default")

        width = 0.5

        match len(dataSets):
            case 1:
                plt.bar(horizontalAxisPos, dataSets[0], width, label=dataLabels[0], color="b")
            case 2:
                width /= 2
                plt.bar(horizontalAxisPos - width/2, dataSets[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos + width/2, dataSets[1], width, label=dataLabels[1], color="r")
            case 3:
                width /= 3
                plt.bar(horizontalAxisPos - width, dataSets[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos, dataSets[1], width, label=dataLabels[1], color="r")
                plt.bar(horizontalAxisPos + width, dataSets[2], width, label=dataLabels[2], color="g")
            case 4:
                width /= 4
                plt.bar(horizontalAxisPos - width - width/2, dataSets[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos - width/2, dataSets[1], width, label=dataLabels[1], color="r")
                plt.bar(horizontalAxisPos + width/2, dataSets[2], width, label=dataLabels[2], color="g")
                plt.bar(horizontalAxisPos + width + width/2, dataSets[3], width, label=dataLabels[3], color="y")

            case 5:
                width /= 5
                plt.bar(horizontalAxisPos - width*2, dataSets[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos - width, dataSets[1], width, label=dataLabels[1], color="r")
                plt.bar(horizontalAxisPos, dataSets[2], width, label=dataLabels[2], color="g")
                plt.bar(horizontalAxisPos + width, dataSets[3], width, label=dataLabels[3], color="y")
                plt.bar(horizontalAxisPos + width*2, dataSets[4], width, label=dataLabels[4], color="k")

            case _:
                raise ValueError("Invalid parameter: dataSets; too many datasets; max is five")

        plt.legend()
        plt.show()

    @staticmethod
    def ConvertCountsToProbabilities(counts: dict) -> list[float]:
        """
        Converts counts, returned from Qiskit.result.getCounts() or Pennylane.counts(), to probabilities which the ProbabilitesToBarChart function can use.
        
        Args:
            counts (dict): Counts dictionary
            shots (int): Number of shots
        Returns:
            list[float]: List of probabilities
        Raises:
            Exception: Invalid parameter: counts; empty dictionary
            Exception: Invalid parameter: counts; must be a dictionary
            Exception: Invalid parameter: counts; is unordered, please call CleanCounts() first

        """

        if len(counts) == 0:
            raise Exception("Invalid parameter: counts; empty dictionary")
        if not isinstance(counts, dict):
            raise Exception("Invalid parameter: counts; must be a dictionary")
        if not QuantumUtils.ToDecimal(list(counts.items())[0][0]) == 0:
            raise Exception("Invalid parameter: counts; is unordered, please call CleanCounts() first")

        totalShots: int = sum(counts.values())
        newCounts: list[float] = []

        for key, value in counts.items():
            newCounts.append(value / totalShots)

        return newCounts

    
    @staticmethod
    def CleanCounts(counts: dict) -> dict:
        """
        Orders and adds all possible quantum states to the counts dictionary.

        Qiskit `job.result.get_counts()` returns a dictionary with the keys as strings and values as integers; function can also be used by Pennylane counts
        However, these are unordered and do not cover all quantum states, use this function to order and add all possible quantum states to the dictionary.
        """

        if not isinstance(counts, dict):
            raise ValueError("Parameter invalid: Counts must be a dictionary.")

        firstKey = next(iter(counts))
        newDict: dict = {}

        numQubits: int = len(firstKey)
        for quantumState in range(2**numQubits):
            # See if the quantum state is already in the dictionary
            quantumStateStr = QuantumUtils.ToBinary(numQubits, quantumState)
            count = counts.get(quantumStateStr)
            if count is None:
                count = 0
            newItem = {QuantumUtils.ToBinary(numQubits, quantumState): count}
            newDict.update(newItem)
        return newDict

    @staticmethod
    def ProbabilitesToBarChart(probabilities: list | list[list], dataLabels: str | list[str], title: str = "Probabilities of quantum circuit", verticalLog: bool = False):
        """
        Creates and prints a bar chart of the probabilities of a quantum circuit. 
        Accepts multiple datasets, but only up to five datasets.

        Args:
            probabilities (list | list[list]): List of probabilities or list of list with probabilities
            dataLabels (str | list[str]): String label for the data or list of labels for the data
            title (str): Title of the graph
            verticalLog (bool): If True, vertical axis will use logaritmic scale, if False, vertical axis will linear scale
        """
        # First convert variable to correct types, if provided list make it into list[list]
        if not isinstance(probabilities[0], list) and not isinstance(probabilities[0], np.ndarray):
            probabilities = [probabilities]
            if len(probabilities[0]) == 0:
                raise ValueError("Parameter invalid: probabilities; empty list")
        # Probabilities is a list of lists, so we need to check if it is empty
        elif any(len(prob) == 0 for prob in probabilities):
            raise ValueError("Parameter invalid: probabilities; some list is empty")

        # dataLabels is a string, make it into list[str]
        if isinstance(dataLabels, str):
            dataLabels = [dataLabels]
        # dataLabels is already a list, so we need to check if it is empty
        elif len(dataLabels) == 0:
            raise ValueError("Parameter invalid: dataLabels; empty list")
        
        if len(dataLabels) != len(probabilities):
            raise ValueError("Parameter invalid: dataLabels and probabilities; datasets must be equal in size")

        qStateLabels = QuantumUtils.CreateQuantumStateLabels(QuantumUtils.GetNumQubits(len(probabilities[0])))
        horizontalAxisPos = np.arange(len(qStateLabels))
        width = 0.8

        plt.style.use("default")
        plt.figure(figsize=(20, 4))
        plt.xlabel("Quantum states")
        plt.ylabel("State occurance percentage")
        plt.title(title)
        plt.xticks(horizontalAxisPos, qStateLabels, rotation=90)
        if verticalLog:
            plt.yscale("log")

            
        match len(dataLabels):
            case 1:
                plt.bar(horizontalAxisPos, probabilities[0], width=width,  label=dataLabels[0], color="b")
            case 2:
                width /= 2
                plt.bar(horizontalAxisPos - width / 2, probabilities[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos + width / 2, probabilities[1], width, label=dataLabels[1], color="r")
            case 3:
                width /= 3
                plt.bar(horizontalAxisPos - width, probabilities[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos, probabilities[1], width, label=dataLabels[1], color="r")
                plt.bar(horizontalAxisPos + width, probabilities[2], width, label=dataLabels[2], color="g")
            case 4:
                width /= 4
                plt.bar(horizontalAxisPos - width - width/2, probabilities[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos - width/2, probabilities[1], width, label=dataLabels[1], color="r")
                plt.bar(horizontalAxisPos + width/2, probabilities[2], width, label=dataLabels[2], color="g")
                plt.bar(horizontalAxisPos + width + width/2, probabilities[3], width, label=dataLabels[3], color="y")
            case 5:
                width /= 5
                plt.bar(horizontalAxisPos - width*2, probabilities[0], width, label=dataLabels[0], color="b")
                plt.bar(horizontalAxisPos - width, probabilities[1], width, label=dataLabels[1], color="r")
                plt.bar(horizontalAxisPos, probabilities[2], width, label=dataLabels[2], color="g")
                plt.bar(horizontalAxisPos + width, probabilities[3], width, label=dataLabels[3], color="y")
                plt.bar(horizontalAxisPos + width*2, probabilities[4], width, label=dataLabels[4], color="k")
            case _:
                raise ValueError("Parameters invalid: dataLabels and probabilities; too many datasets; max is five")

        plt.legend()
        plt.show()