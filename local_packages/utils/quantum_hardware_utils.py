from enum import Enum
import pennylane as qml
from functools import partial

class QuantumHardwareUtils:
    """
    Utility class for quantum hardware, last updated at 25-04-2025
    """
    # Source: https://quantum.ibm.com/services/resources?system=ibm_torino
    ibmHeronR1GateSet = {qml.CZ, qml.Identity, qml.RX, qml.RZ, qml.IsingZZ, qml.SX, qml.PauliX}
    # Source: https://quantum.ibm.com/services/resources?system=ibm_aachen
    ibmHeronR2GateSet = {qml.CZ, qml.Identity, qml.RX, qml.RZ, qml.IsingZZ, qml.SX, qml.PauliX}
    # Source: https://quantum.ibm.com/services/resources?system=ibm_sherbrooke
    ibmEagleR3GateSet = {qml.ECR, qml.Identity, qml.RZ, qml.SX, qml.PauliX}

    # Spec sheet of Google's Willow: https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf
    # google re-uses it architecture and supportes natively the following gates: source: https://quantumai.google/cirq/google/devices
    googleWillowGateSet = {"X", "Y", "Z", "RX", "RY", "RZ", "CZ", "I"}

    class QuantumHardware(Enum):
        GOOGLE_WILLOW = 0
        IBM_HERON_R1 = 1
        IBM_HERON_R2 = 2
        IBM_EAGLE_R3 = 3

    @staticmethod
    def GetNativeGateSet(quantumHw: QuantumHardware) -> set:
        """
        Returns the native gate set of the specified quantum hardware.
        """
        match quantumHw:
            case QuantumHardwareUtils.QuantumHardware.GOOGLE_WILLOW:
                return QuantumHardwareUtils.googleWillowGateSet
            case QuantumHardwareUtils.QuantumHardware.IBM_HERON_R1:
                return QuantumHardwareUtils.ibmHeronR1GateSet
            case QuantumHardwareUtils.QuantumHardware.IBM_HERON_R2:
                return QuantumHardwareUtils.ibmHeronR2GateSet
            case QuantumHardwareUtils.QuantumHardware.IBM_EAGLE_R3:
                return QuantumHardwareUtils.ibmEagleR3GateSet
            case _:
                raise Exception("unsupported quantum hardware specified.")
            
    @staticmethod
    def Decompose(circuit: callable, hardware: QuantumHardware) -> callable:
        """
        Decomposes a quantum circuit into native gates.
        Args:
            circuit (callable): The quantum circuit to decompose.
            hardware (QuantumHardware): The quantum hardware to use for decomposition.

        Returns:
            callable: The decomposed quantum circuit.

        Raises:
            Exception: If the specified quantum hardware is not supported.
        """
        return qml.transforms.decompose(circuit, gate_set=QuantumHardwareUtils.GetNativeGateSet(hardware))