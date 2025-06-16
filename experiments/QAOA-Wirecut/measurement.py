from pennylane.operation import Operation

import pennylane as qml

class PauliCallback:
    def __init__(self, PauliGate: str, qubit: int):
        
        if PauliGate not in ["x", "y", "z"]:
            raise ValueError(f"Invalid Pauli gate; received: {PauliGate}, must be: x, y, or z")
        
        self.PauliGate: str = PauliGate
        self.qubit: int = qubit

    def __str__(self) -> str:
        return f"PauliCallback: PauliGate={self.PauliGate}, on qubit={self.qubit}"
    
    def ToQasm(self) -> str:
        return f"{self.PauliGate} q[{self.qubit}];\n"

class Measurement:
    """
    Data object to store information about a measurement.
    """
    def __init__(self, wire: int, bit: int = -1, reset: bool = False, callback: PauliCallback = None):
        """
        Measurement constructor.
        Args:
            wire (int): Wire index to measure.
            bit (int): Bit index to store the measurement result. If bit < 0 measurement value is discarded.
            reset (bool): Whether to reset the wire after measurement.
            callback (PauliCallback): Optional callback for additional single qubit gates.
        """
        if wire < 0:
            raise ValueError(f"Wire index out of range; received: {wire}")    
        
        # if the measurement needs to be discarded and the qubit reset, than temporarily store the measurement value for the reset
        if bit < 0:
            # Bit parameter not defined, use ancillary (a) classical register
            self.register: chr = 'a'
            self.bit = 0
        else:
            # Bit parameter defined, use (c) classical register
            self.register: chr = 'c'
            self.bit: int = bit
        
        self.wire: int = wire
        self.reset: bool = reset
        self.callback: PauliCallback = callback

    def __str__(self) -> str:
        return f"Measurement: wire={self.wire}, bit={self.bit}, reset={self.reset}"
    
    def ToQasm(self) -> str:
        qasmStr: str = f"measure q[{self.wire}] -> {self.register}[{self.bit}];\n"
        
        if self.reset == True or self.callback is not None:
            qasmStr += f"if ({self.register}[{self.bit}]) {{\n"
            if self.reset:
                qasmStr += f"reset q[{self.wire}];\n"
                
            if self.callback is not None:
                qasmStr += f"{self.callback.ToQasm()}"
            qasmStr += f"}}\n"
        return qasmStr

class MeasurementGate(Operation):
    """
    Custom measurement gate for PennyLane, implements the 'Pennylane Operation' interface.
    """
    num_params = 2 # bit and reset
    num_wires = 1
    par_domain = "R"

    def __init__(self, bit: int, reset: bool, wires: int):
        super().__init__(bit, reset, wires=wires, id="MeasurementGate")

    @staticmethod
    def compute_decomposition(bit, reset, wires):
        """
        When decomposed to QASM 2.0, the measurement gate looks like an Identity gate, but will later be transpiled to QASM 3.0 using the ToQasm3 function
        """
        return [qml.Identity(wires=wires)]