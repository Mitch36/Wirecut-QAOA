
from quantum_utils import QuantumUtils as qu


# Good cases
def test_DifferenceSumPositive():
    probsA = [1, 2, 3, 4]
    probsB = [5, 6, 7, 8]
    answer = qu.DifferenceSum(probsA, probsB)
    assert answer == 16


def test_DifferenceSumPositiveReversedInput():
    probsB = [5, 6, 7, 8]
    probsA = [1, 2, 3, 4]
    answer = qu.DifferenceSum(probsA, probsB)
    assert answer == 16

def test_DifferenceSumNegative():
    probsA = [-1, -2, -3, -4]
    probsB = [3, 2, 1, 0]
    answer = qu.DifferenceSum(probsA, probsB)
    assert answer == 16

def test_DifferenceSumBothNegative():
    probsA = [-1, -2, -3, -4]
    probsB = [-5, -6, -7, -8]
    answer = qu.DifferenceSum(probsA, probsB)
    assert answer == 16

# Bad cases
def test_DifferenceSumEmptyInputA():
    try:
        probsA = []
        probsB = [1]
        answer = qu.DifferenceSum(probsA, probsB)
    except Exception as e:
        assert str(e) == "Parameter invalid: contentA, is empty list"

def test_DifferenceSumEmptyInputB():
    try:
        probsA = [1]
        probsB = []
        answer = qu.DifferenceSum(probsA, probsB)
    except Exception as e:
        assert str(e) == "Parameter invalid: contentB, is empty list"