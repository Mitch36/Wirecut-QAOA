from enum import Enum, auto

class QuantumBackEnd(Enum):
    
    PENNYLANE = "Pennylane"
    QISKIT_AER = "Qiskit-Aer"
    QISKIT_AER_IBM_SHERBROOKE = "Qiskit-Aer-IBM-Sherbrooke"
    QISKIT_AER_IBM_BRISBANE = "Qiskit-Aer-IBM-Brisbane"
    IBMQ_SHERBROOKE = "IBMQ-Sherbrooke"
    IBMQ_BRISBANE = "IBMQ-Brisbane"