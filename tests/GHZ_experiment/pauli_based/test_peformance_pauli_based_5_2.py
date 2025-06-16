from GHZ_pauli_wire_cut import GHZPauliWireCut
from quantum_wire_cutting import QuantumWireCutUtils as qu

# Test the performance of the Pauli-based approach with parameters
# qubits = 5; cuts = 2;

ERROR_THRESHOLD = 0.1

def test_performance_pauli_based_5_2_1000():

    passedCount = 0
    failedCount = 0
    for i in range(100):
        exp = GHZPauliWireCut(5, 2, 1000)
        orgProbs = exp.RunOriginialCircuit()
        shots = exp.DoSubCircuitMeasurements()
        cutProbs = exp.ConstructFullProbabilites()
        if qu.EvaluateProbs(orgProbs, cutProbs, ERROR_THRESHOLD):
            passedCount += 1
        else:
            failedCount += 1

    assert passedCount > failedCount


def test_performance_pauli_based_5_2_500():
    passedCount = 0
    failedCount = 0
    for i in range(100):
        exp = GHZPauliWireCut(5, 2, 500)
        orgProbs = exp.RunOriginialCircuit()
        shots = exp.DoSubCircuitMeasurements()
        cutProbs = exp.ConstructFullProbabilites()
        if qu.EvaluateProbs(orgProbs, cutProbs, ERROR_THRESHOLD):
            passedCount += 1
        else:
            failedCount += 1

    assert not passedCount > failedCount


def test_performance_pauli_based_5_2_250():
    passedCount = 0
    failedCount = 0
    for i in range(100):
        exp = GHZPauliWireCut(5, 2, 250)
        orgProbs = exp.RunOriginialCircuit()
        shots = exp.DoSubCircuitMeasurements()
        cutProbs = exp.ConstructFullProbabilites()
        if qu.EvaluateProbs(orgProbs, cutProbs, ERROR_THRESHOLD):
            passedCount += 1
        else:
            failedCount += 1

    assert not passedCount > failedCount


def test_performance_pauli_based_5_2_100():
    passedCount = 0
    failedCount = 0
    for i in range(100):
        exp = GHZPauliWireCut(5, 2, 100)
        orgProbs = exp.RunOriginialCircuit()
        shots = exp.DoSubCircuitMeasurements()
        cutProbs = exp.ConstructFullProbabilites()
        if qu.EvaluateProbs(orgProbs, cutProbs, ERROR_THRESHOLD):
            passedCount += 1
        else:
            failedCount += 1

    assert not passedCount > failedCount
