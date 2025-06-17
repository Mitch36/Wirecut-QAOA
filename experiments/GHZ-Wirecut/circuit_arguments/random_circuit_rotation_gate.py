import random
from math import pi

import pennylane as qml
from pennylane.numpy import random
from circuit_argument_interface import CircuitArgumentInterface


class RandomCircuitRotationGate(CircuitArgumentInterface):
    def __init__(self, numQubits: int = -1, overwriteQubit: int = -1, overwriteGate: str = None):
        # 1. Generate a random qubit index between 0 and numQubits - 1 or if overwrite qubit provided, use that one
        self.numQubits = numQubits
        if overwriteQubit == -1 and numQubits != -1:
            self.qubit = random.randint(0, numQubits)
            if self.qubit == numQubits:
                self.qubit = numQubits - 1
        elif overwriteQubit != -1 and numQubits == -1:
            self.qubit = overwriteQubit
        else:
            raise ValueError(
                "Parameter invalid: numQubits; numQubits cannot be -1 if overwriteQubit is -1")
        
        # 2. Generate a random gate type
        if overwriteGate is not None:
            if overwriteGate == "U3":
                self.gate = "U3"
                self.theta = random.uniform(0, pi)
                self.delta = random.uniform(0, 2*pi)
            elif overwriteGate not in ["RX", "RY", "RZ"]:
                raise ValueError("Invalid gate type; must be: RX, RY, RZ")
            self.gate = overwriteGate
        else:
            self.gate = random.choice(["RX", "RY", "RZ"])
        
        # 3. Generate a random angle between 0 and pi
        self.phi = random.uniform(0, 2*pi)

    def Apply(self, overwriteQubit: int = -1):
        qubit = self.qubit
        if overwriteQubit != -1:
            qubit = overwriteQubit

        match self.gate:
            case "RX":
                qml.RX(self.phi, wires=[qubit])
            case "RY":
                qml.RY(self.phi, wires=[qubit])
            case "RZ":
                qml.RZ(self.phi, wires=[qubit])
            case "U3":
                qml.U3(self.phi, self.theta, self.delta, wires=[qubit])
            case _:
                raise ValueError("Invalid gate type")
            
    def Get(self) -> dict:
        """
        Returns the gate type and qubit index as dictionary

        Returns:
            dict: A dictionary containing the gate type and qubit index.
        """
        if self.gate == "U3":
            return dict(gate=self.gate, wires=[self.qubit], phi=self.phi, theta=self.theta, delta=self.delta)
        return dict(gate=self.gate, wires=[self.qubit], phi=self.phi)

    def __str__(self) -> str:
        return (
            "RandomCircuitRotationGate: "
            + str(self.gate)
            + " with angle: "
            + str(self.phi)
            + " on qubit: "
            + str(self.qubit)
        )
