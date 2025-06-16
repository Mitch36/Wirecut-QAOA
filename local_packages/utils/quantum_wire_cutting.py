
from typing import Union

from pennylane import numpy as np
from pennylane.numpy import random
import pennylane as qml
from pennylane.tape import QuantumScript, QuantumTape
from quantum_utils import QuantumUtils as qu
from circuit_arguments.random_circuit_rotation_gate import RandomCircuitRotationGate

class QuantumWireCutUtils:

    @staticmethod
    def KroneckerProductSum(p1_list, p2_list):
        """
        Calculates the sum of Kronecker products: (1/2) * Σ(i=1 to 4) p1_i ⊗ p2_i

        Args:
            p1_list: A list of 4 NumPy matrices (2x2) representing p1_i.
            p2_list: A list of 4 NumPy matrices (2x2) representing p2_i.

        Returns:
            A NumPy matrix (4x4) representing the result.
        """

        if p1_list is None or p2_list is None:
            raise ValueError("Input lists must not be None.")

        # if len(p1_list) != 4 or len(p2_list) != 4:
        #     raise ValueError("Input lists must contain 4 matrices each.")

        result = np.zeros((4, 4), dtype=complex)

        for i in range(4):
            p1_i = p1_list[i]
            p2_i = p2_list[i]
            result += np.kron(p1_i, p2_i)

        return (1 / 2) * result

    @staticmethod
    def ComplexToNormal(complex_number: Union[float, list]) -> Union[float, list]:
        """
        Converts one or multiple complex number(s) to normal number(s).

        Args:
            complex_numbers: One or multiple complex numbers.

        Returns:
            One or multiple normal numbers.

        Raises:
            ValueError: Input must not be None.
            ValueError: Input list must not be empty.
            ValueError: Input must be either a float or a list.
        """
        if complex_number is None:
            raise ValueError("Input must not be None.")

        if isinstance(complex_number, list):
            if complex_number == []:
                raise ValueError("Input list must not be empty.")
            for i in range(len(complex_number)):
                complex_number[i] = np.float64((abs(complex_number[i])))
            return complex_number
        elif isinstance(complex_number, float):
            return abs(complex_number)
        else:
            raise ValueError("Input must be either a float or a list.")

    @staticmethod
    def EvaluateProbs(probA: list, probB: list, threshold: float = 0.05) -> bool:
        """
        Evaluates the probabilites of two different datasets using a defined threshold.

        Args:
            probA (list): Dataset A
            probB (list): Dataset B
            threshold (float): Threshold which defines what the return value will be

        Returns:
            True if the difference between the datasets for each quantum state is smaller than the threshold parameter
            False if the difference between the datasets for a quantum state is greater than or equal to the threshold parameter
        """

        if len(probA) == 0:
            raise Exception("Parameter invalid: probA, is empty list")

        if len(probB) == 0:
            raise Exception("Parameter invalid: probB, is empty list")

        if len(probA) != len(probB):
            raise Exception("Parameter invalid: probA and probB, datasets must be equal in size")

        if threshold <= 0:
            raise Exception("Parameter invalid: threshold, must be greater than 0")

        for A, B in zip(probA, probB):
            diff = A - B
            if diff < 0:
                diff *= -1
            if diff >= threshold:
                print("diff: ", diff)
                return False
        return True

    @staticmethod
    def RandomCliffordCircuit(wires: list, depth: int) -> list:
        """ "
        Creates a random Clifford circuit with a given number of qubits and depth.

        Args:
            wires (list): List of qubits to apply the gates on.
            depth (int): Depth of the circuit.

        Returns:
            list: List of operations in the circuit as dictionaries with values for gate and wires.
        """

        if depth < 1:
            raise ValueError("Parameter invalid; Depth must be greater than 0.")

        if len(wires) > 1:
            clifford_gates = ['H', 'S', 'X', 'CNOT', 'Y', 'Z']
            # clifford_gates = ['H', 'S', 'CNOT']

        else:
            # If only one wire, remove CNOT from the list of gates
            clifford_gates = ['H', 'S', 'X', 'Y', 'Z']
            # clifford_gates = ['H', 'S']

        operations = []

        for wire in wires:
            # Choose a gate for each wire
            for i in range(depth):
                gate = random.choice(clifford_gates)

                if gate == "CNOT":
                    # Choose random control and target wire
                    controlWire = random.choice(wires)
                    targetWire = random.choice(wires)
                    while controlWire == targetWire:
                        controlWire = random.choice(wires)
                        targetWire = random.choice(wires)
                    operations.append(dict(gate=gate, wires=[controlWire, targetWire]))
                else:
                    operations.append(dict(gate=gate, wires=wire))
        return operations
        
    @staticmethod
    def RandomRotationGate(wire: int) -> list:
        """
        Returns a list of operations with one random rotation gate applied to a given wire.
        Args:
            wire (int): Wire to apply the rotation gate to.
        Returns:
            list: List of operations with the rotation gate applied.
        """
        operations = []
        rotGate = RandomCircuitRotationGate(overwriteQubit=wire)
        operations.append(rotGate.Get())
        return operations
    
    @staticmethod
    def RandomRotations(wire: int) -> list:
        """
        Returns a list of operations with a U3 operation which applies a random rotation on all three axis.
        Args:
            wires (int): Wire to apply the rotation gate to.
        Returns:
            list: List of operations with the rotation gate applied.
        """
        operations = []
        u3Gate = RandomCircuitRotationGate(overwriteQubit=wire, overwriteGate="U3")
        operations.append(u3Gate.Get())
        return operations
        

    def WireShift(operations: list, shift: int) -> list:
        """
        Shifts the wires of the operations by a given amount. example: CNOT(0, 1) with shift 2 becomes CNOT(2, 3)

        Args:
            operations (list): List of operations to shift.
            shift (int): Amount to shift the wires.

        Returns:
            list: List of operations with shifted wires.
        """
        if not isinstance(operations, list):
            raise ValueError("Parameter invalid: operations, must be a list")

        for op in operations:
            wires = op.get("wires")
            for i in range(len(wires)): # TODO, fix this error, since not all 'wires' can be arrays but also int
                wires[i] += shift
        return operations
    
    def WireMove(operations: list, wire: int) -> list:
        """
        Moves all the wires of the operations to a given wire. example: X(0) with wire 0 to wire 2 becomes X(2)

        Args:
            operations (list): List of operations to shift.
            wire (int): Wire to move the operations to.

        Returns:
            list: List of operations with moved wires.
        """
        if not isinstance(operations, list):
            raise ValueError("Parameter invalid: operations, must be a list")

        for op in operations:
            wires = op.get("wires")
            for i in range(len(wires)): # TODO, fix this error, since not all 'wires' can be arrays but also int
                wires[i] = wire
        return operations
    
    def WireStartFromReversed(operations: list, startFromWire: int) -> list:
        """
        Moves the wires of the operations to a given wire. example: X(2), Z(1) with end from 3 becomes X(1), Z(2) OR X(1), Z(2) with end from 3 becomes X(3), Z(4)

        Args:
            operations (list): List of operations to shift.
            wire (int): Wire to move the operations to.

        Returns:
            list: List of operations with moved wires.
        """
        if not isinstance(operations, list):
            raise ValueError("Parameter invalid: operations, must be a list")

        cealing: int = startFromWire + 1

        for op in operations:
            wires = op.get("wires")
            if isinstance(wires, int):
                wires = cealing - wires
                op["wires"] = wires
            elif isinstance(wires, list):
                for i in range(len(wires)):
                    wires[i] = cealing - wires[i]
                op["wires"] = wires
            else:
                raise ValueError("Parameter invalid: operations, wires must be int or list")
        return operations
    
    def WireStartFrom(operations: list, startFromWire: int) -> list:
        """
        Moves the wires of the operations to a given wire. example: X(1), Z(3) with start from 5 becomes X(3), Z(5) OR X(1), Z(2) with start from 3 becomes X(2), Z(3)

        Args:
            operations (list): List of operations to shift.
            wire (int): Wire to move the operations to.

        Returns:
            list: List of operations with moved wires.
        """
        if not isinstance(operations, list):
            raise ValueError("Parameter invalid: operations, must be a list")

        smallestWire: int = 9999
        for op in operations:
            wires = op.get("wires")
            if isinstance(wires, int):
                if wires < smallestWire:
                    smallestWire = wires
            else:
                for i in range(len(wires)):
                    if wires[i] < smallestWire:
                        smallestWire = wires[i]

        shift = smallestWire - startFromWire

        for op in operations:
            wires = op.get("wires")
            if isinstance(wires, int):
                op["wires"] = wires - shift
            else:
                for i in range(len(wires)):
                    wires[i] = wires[i] - shift
                op["wires"] = wires
        return operations
    
    @staticmethod
    def ApplyCircuit(operations: list, adjoint: bool = False) -> QuantumTape:
        """
        Applies a list of operations to a quantum circuit. Notice this function can only be used within the PennyLane QNode decorator.

        Args:
            operations (list): List of operations to apply.
            adjoint (bool): If True, applies the adjoint of the operations.
        """

        if len(operations) == 0:
            return

        instructions = []
        for op in operations:
            match op["gate"]:
                case "H":
                    instructions.append(qml.Hadamard(wires=op["wires"]))
                case "X":
                    instructions.append(qml.PauliX(wires=op["wires"]))
                case "Y":
                    instructions.append(qml.PauliY(wires=op["wires"]))
                case "Z":
                    instructions.append(qml.PauliZ(wires=op["wires"]))
                case "S":
                    instructions.append(qml.S(wires=op["wires"]))
                case "CNOT":
                    instructions.append(qml.CNOT(wires=op["wires"]))
                case "RX":
                    instructions.append(qml.RX(op["phi"], wires=op["wires"]))  
                case "RY":
                    instructions.append(qml.RY(op["phi"], wires=op["wires"]))
                case "RZ":
                    instructions.append(qml.RZ(op["phi"], wires=op["wires"])) 
                case "U3":
                    instructions.append(qml.U3(op["theta"], op["phi"], op["delta"], wires=op["wires"])) 
                case _:
                    raise ValueError(f"Gate unsupported: {op['gate']}.")
        if adjoint:
            adjointOps = []
            for op in reversed(instructions):
                adjointOps.append(qml.adjoint(op))
            return QuantumTape(ops=adjointOps)
        return QuantumTape(ops=operations)

    @staticmethod
    def PrepareBasis(wires: int | list[int], basis: str):
        """
        Prepares a qubit, or multiple qubits, in a given basis. Note that this function can only be used with PennyLane QNode decorator.

        Args:
            wire (int): The wire on which the qubit is prepared.
            basis (str): The basis in which the qubit is prepared.

        Raises:
            ValueError: The basis is invalid.
        """

        if isinstance(wires, int):
            wires = [wires]

        for wire in wires:
            match basis:
                case "0":
                    pass
                case "1":
                    qml.PauliX(wires=wire)
                case "+":
                    qml.Hadamard(wires=wire)
                case "-":
                    qml.Hadamard(wires=wire)
                    qml.PauliZ(wires=wire)
                case "i" | "+i":
                    qml.Hadamard(wires=wire)
                    qml.S(wires=wire)
                case "-i":
                    qml.Hadamard(wires=wire)
                    qml.adjoint(qml.S(wires=wire))
                case _:
                    raise ValueError(f"Invalid parameter; basis: {basis} does not exist.")

    @staticmethod
    def PrepareRandomBasis(wires: int | list[int]):
        """
        Prepares a qubit in a random basis. Note that this function can only be used with PennyLane QNode decorator.

        Args:
            wire (int) or (list[int]): The wire(s) on which the qubit(s) are prepared.
        """
        basis: list[str] = ['0', '1', '+', '+i']

        if isinstance(wires, int):
            wires = [wires]
        
        basis: str = random.choice(basis)
        for wire in wires:
            QuantumWireCutUtils.PrepareBasis(wire, basis)