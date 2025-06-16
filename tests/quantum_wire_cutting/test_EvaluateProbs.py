from quantum_wire_cutting import QuantumWireCutUtils as qu


# Good cases
def test_EvaulateProbsTrue():
    probsA = [0.5, 0.5]
    probsB = [0.4, 0.4]
    assert qu.EvaluateProbs(probsA, probsB, 0.11)


def test_EvaulateProbsFalse():
    probsA = [0.5, 0.5]
    probsB = [0.4, 0.4]
    assert not qu.EvaluateProbs(probsA, probsB, 0.09)


# Bad cases
def test_EvaulateProbsEmptyListA():
    try:
        qu.EvaluateProbs([], [0.5, 0.5])
    except Exception as e:
        assert str(e) == "Parameter invalid: probA, is empty list"


def test_EvaulateProbsEmptyListB():
    try:
        qu.EvaluateProbs([0.5, 0.5], [])
    except Exception as e:
        assert str(e) == "Parameter invalid: probB, is empty list"


def test_EvaulateProbsSizeDifference():
    try:
        qu.EvaluateProbs([0.5, 0.5], [0.5])
    except Exception as e:
        assert str(e) == "Parameter invalid: probA and probB, datasets must be equal in size"


def test_EvaulateProbsThresholdZero():
    try:
        qu.EvaluateProbs([0.5, 0.5], [0.5, 0.5], 0)
    except Exception as e:
        assert str(e) == "Parameter invalid: threshold, must be greater than 0"
