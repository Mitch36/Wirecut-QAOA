from quantum_utils import QuantumUtils as qu
import numpy as np

# Good cases
def test_Normalize():
    orgProbs = [0.1, 0.2, 0.3, 0.4]
    invalidProbs = [0.2, 0.4, 0.6, 0.8]
    
    assert np.sum(orgProbs) == 1
    assert np.sum(invalidProbs) == 2

    normalized = qu.Normalize(invalidProbs)

    assert np.sum(normalized) == 1

def test_NormalizeSmallNumbers():
    orgProbs = [0.1, 0.2, 0.3, 0.4]
    invalidProbs = [0.05, 0.1, 0.15, 0.2]
    
    assert np.sum(orgProbs) == 1
    assert np.sum(invalidProbs) == 0.5

    normalized = qu.Normalize(invalidProbs)

    assert np.sum(normalized) == 1

def test_NormalizeLargeNumbers():
    orgProbs = [0.1, 0.2, 0.3, 0.4]
    invalidProbs = [1000, 2000, 3000, 4000]
    
    assert np.sum(orgProbs) == 1
    assert np.sum(invalidProbs) == 10000

    normalized = qu.Normalize(invalidProbs)

    assert np.sum(normalized) == 1

def test_NormalizeFloats():
    invalidProbs = [1.74302470134, 0.730141201901, 0.037147321034, 0.000438104]

    normalized = qu.Normalize(invalidProbs)
    sum = np.sum(normalized)

    assert sum == 1

def test_NormalizeNegativeFloats():
    invalidProbs = [-1.74302470134, -0.730141201901, -0.037147321034, -0.000438104]

    normalized = qu.Normalize(invalidProbs)
    assert np.sum(normalized)

def test_NormalizeMixedFloats():
    invalidProbs = [-1.74302470134, 0.730141201901, -0.037147321034, 0.000438104]

    normalized = qu.Normalize(invalidProbs)
    assert np.sum(normalized) == 1 

# Bad cases
def test_NormalizeEmptyInput():
    try:
        probs = []
        qu.Normalize(probs)
    except Exception as e:
        assert str(e) == "Parameter invalid: list; empty list"
