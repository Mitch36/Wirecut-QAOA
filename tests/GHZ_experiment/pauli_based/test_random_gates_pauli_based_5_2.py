from circuit_configurations.circuit_random_gates_configuration import (
    RandomCircuitGatesConfiguration,
)
from GHZ_pauli_wire_cut import GHZPauliWireCut
from quantum_wire_cutting import QuantumWireCutUtils as qu


def test_random_gates_pauli_based_5_2_100():
    passedCount = 0
    failedCount = 0

    NUM_QUBITS = 5
    for i in range(100):

        config = RandomCircuitGatesConfiguration(NUM_QUBITS, NUM_QUBITS)
        exp = GHZPauliWireCut(numQubits=NUM_QUBITS, numCuts=2, shotsBudget=2000, config=config)
        orgProbs = exp.RunOriginialCircuit()
        shots = exp.DoSubCircuitMeasurements()
        cutProbs = exp.ConstructFullProbabilites()
        if qu.EvaluateProbs(orgProbs, cutProbs, threshold=0.1):
            passedCount += 1
        else:
            failedCount += 1

    print(f"Passed: {passedCount}, Failed: {failedCount}")
    assert failedCount < 10
