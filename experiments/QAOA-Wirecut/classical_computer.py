from measurement import Measurement, MeasurementGate, PauliCallback

class InitialMeasurement:
    def __init__(self, wires: list[int], measurements: list, cliffCirc: list, layer: int):
        self.wires: list[int] = wires
        self.measurements: list = measurements
        self.cliffCirc: list = cliffCirc
        self.layer: int = layer

    def __str__(self) -> str:
        outputStr: str = f"Measurement: \n"
        outputStr += f"Wires: {self.wires}\n"
        outputStr += f"Measurements: {self.measurements}\n"
        outputStr += f"Clifford Circuit: {self.cliffCirc}\n"
        outputStr += f"Executed in layer: {self.layer}\n"
        return outputStr

class ClassicalComputer:
    def __init__(self):
        self.numBits = "To be defined by measurement function call(s)"
        self.measurements: list[Measurement] = []
        self.initMeasurements: list[InitialMeasurement] = []

    def Measure(self, wire: int, bit: int = -1, reset: bool = False, callback: PauliCallback = None) -> None:
        """"
        (Mid) circuit measurement of a wire with the classical computer.

        Args:
            wire (int): Wire index to measure.
            bit (int): Bit index to store the measurement result. If bit < 0 measurement value is discarded.
            reset (bool): Whether to reset the wire after measurement.
            callback (PauliCallback): Optional callback for additional single qubit gates.
        """
        wire = wire - 1  # Adjust for zero-based indexing
        
        if wire < 0:
            raise ValueError(f"Wire index out of range; received: {wire}")

        self.measurements.append(Measurement(wire, bit, reset, callback))
        MeasurementGate(bit=bit, reset=reset, wires=[wire])

    def AppendMeasure(self, wire: int, reset: bool = False, callback: PauliCallback = None) -> None:
        """"
        (Mid) circuit measurement of a wire with the classical computer, auto appends the measurment to the classical register.

        Args:
            wire (int): Wire index to measure.
            reset (bool): Whether to reset the wire after measurement.
            callback (PauliCallback): Optional callback for additional single qubit gates.
        """

        wire = wire - 1  # Adjust for zero-based indexing

        if wire < 0:
            raise ValueError(f"Wire index out of range; received: {wire}")

        # Find the first free slot in the classical register
        freeBitIndex: int = -1
        if len(self.measurements) == 0:
            freeBitIndex = 0
            self.measurements.append(Measurement(wire, freeBitIndex, reset, callback))
            MeasurementGate(bit=freeBitIndex, reset=reset, wires=[wire])
            return

        for measurement in self.measurements:
            if measurement.bit > freeBitIndex and measurement.register != 'a': # Skip ancillary register measurements
                freeBitIndex = measurement.bit

        freeBitIndex += 1  # Increment to find the next free bit index

        self.measurements.append(Measurement(wire, freeBitIndex, reset, callback))
        MeasurementGate(bit=freeBitIndex, reset=reset, wires=[wire])

    def MeasurePrepare(self, measureWire: int, prepareWire: int, reset: bool = False) -> None:
        """
        (Mid) circuit measurement of a (measureWire) with the classical computer and prepares (prepareWire) based on measurement, auto appends the measurment to the ancillary (a) classical register.
        """

        measureWire = measureWire - 1  # Adjust for zero-based indexing

        if measureWire < 0:
            raise ValueError(f"Wire index out of range; received: {measureWire}")

        if prepareWire < 0:
            raise ValueError(f"Wire index out of range; received: {prepareWire}")

        self.measurements.append(Measurement(wire=measureWire, reset=True, callback=PauliCallback('x', prepareWire)))
        MeasurementGate(bit=measureWire, reset=reset, wires=[prepareWire])

    def StoreMeasurementAndClifford(self, wires: list[int], measurements: list, cliffCirc: list, layer: int) -> None:
        """
        Store the measurement and Clifford operations in the classical computer.
        """
        self.initMeasurements.append(InitialMeasurement(wires, measurements, cliffCirc, layer))

    def StoreMeasurement(self, wires: list[int], measurements: list, layer: int) -> None:
        self.initMeasurements.append(InitialMeasurement(wires, measurements, [], layer=layer))

    def __str__(self) -> str:
        outputStr: str = f"ClassicalComputer: \n"
        for measurement in self.measurements:
            outputStr += str(measurement) + "\n"
        return outputStr