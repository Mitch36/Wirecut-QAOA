from quantum_utils.src.quantum_utils import QuantumUtils as qu


# Good cases
def test_BarChartValidProbabilites():
    exceptionOccured = ""
    try:
        qu.BarChart([0.5, 0.5, 0.5, 0.5])
    except Exception as e:
        exceptionOccured = str(e)
    finally:
        assert exceptionOccured == ""


# Bad cases
def test_BarChartEmptyList():
    try:
        qu.BarChart([])
    except Exception as e:
        assert str(e) == "Invalid parameter: probabilities; empty list"


def test_BarChartInvalidLengthProbabilites():
    try:
        qu.BarChart([0.5, 0.5, 0.5, 0.5, 0.5])
    except Exception as e:
        assert str(e) == "Parameter invalid: numStates; invalid length compared to possible qubits"
