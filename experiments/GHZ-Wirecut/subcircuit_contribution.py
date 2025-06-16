from subcircuit_position import SubcircuitPosition


class SubCircuitContribution:
    """
    Class to represent the contribution of a subcircuit to the total bitstring.
    """

    def __init__(self, startIndex: int, length: int, totalLength: int):
        self.startIndex = startIndex
        self.length = length
        self.endIndex = startIndex + self.length
        self.totalLength = totalLength

    def ToContribution(self, stateStr: str) -> str:
        """
        Translates from a original bit string its contribution
        """
        return stateStr[self.startIndex : self.endIndex]

    def ToString(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            " Contribution: from: "
            + str(self.startIndex)
            + " to: "
            + str(self.endIndex)
            + " length: "
            + str(self.length)
            + " of total: "
            + str(self.totalLength)
            + "; bitstring: "
            + self.ToBitStr()
        )

    def ToBitStr(self) -> str:
        bitStr = ""
        for i in range(self.totalLength):
            if i == self.startIndex:
                bitStr += "["
            if self.startIndex <= i < self.endIndex:
                bitStr += "0"
                if i == self.endIndex - 1:
                    bitStr += "]"
            elif i < self.startIndex or i >= self.endIndex:
                bitStr += "X"
        return bitStr
