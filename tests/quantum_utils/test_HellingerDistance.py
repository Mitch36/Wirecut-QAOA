from quantum_utils import QuantumUtils as qu


# Good cases
def test_HellingerDistance():
    # Explanation:
    # Hellinger distance is a measure of the similarity between two probability distributions.
    # It is defined as the square root of the sum of the squared differences between the
    # square roots of the probabilities of the two distributions.
    # In this case, we have two distributions:
    probsA = [0.5, 0.5]
    probsB = [0.1, 0.9]
    # The Hellinger distance between these two distributions is:
    # H = sqrt(0.5 * (sqrt(0.5) - sqrt(0.1))^2 + 0.5 * (sqrt(0.5) - sqrt(0.9))^2)
    # H = sqrt(0.5 * (0.7071 - 0.3162)^2 + 0.5 * (0.7071 - 0.9487)^2)
    # H = sqrt(0.5 * (0.3909)^2 + 0.5 * (0.2416)^2)
    # H = sqrt(0.5 * 0.1524 + 0.5 * 0.0583)
    # H = sqrt(0.0764 + 0.0292)
    # H = sqrt(0.1056)
    # H = 0.3254

    assert qu.HellingerDistance(probsA, probsB) == 0.3249196962329063

def test_HellingerDistanceCompared():
    # Explanation:
    # The Hellinger distance scales based on the difference between probabilities.
    # In this case, we have three probability distributions:
    probsA = [0.0, 0.0, 1.0]
    probsB = [0.25, 0.25, 0.5]
    probsC = [0.5, 0.0, 0.5]

    # Both share the same amount of difference: 
    # Difference(AB) = 0.25 + 0.25 + 0.5 = 1
    # Difference(AC) = 0.5 + 0.0 + 0.5 = 1
    assert qu.DifferenceSum(probsA, probsB) == qu.DifferenceSum(probsA, probsC)

    # However, the hellinger distance is different since it increases based on the step size; 
    # Therefore, probsC's Hellinger distance is greater than probsB's due the larger step size: 0.5
    # In short: difference might be the same, but the Hellinger distance is not

    assert qu.HellingerDistance(probsA, probsB) < qu.HellingerDistance(probsA, probsC)

# Bad cases
def test_HellingerDistanceEmptyInputA():
    try:
        probsA = []
        probsB = [1]
        answer = qu.HellingerDistance(probsA, probsB)
    except Exception as e:
        assert str(e) == "Both probability distributions must be non-empty."

def test_HellingerDistanceEmptyInputB():
    try:
        probsA = [1]
        probsB = []
        answer = qu.HellingerDistance(probsA, probsB)
    except Exception as e:
        assert str(e) == "Both probability distributions must be non-empty."

def test_HellingerDistanceNegativeNumbers():
    try:
        probsA = [1.1, -0.1]
        probsB = [0.1, 0.9]
        answer = qu.HellingerDistance(probsA, probsB)
    except Exception as e:
        assert str(e) == "Probability values must be non-negative."

def test_HellingerDistanceInvalidProbabilities():
    try:
        probsA = [0.1, 0.1] # Total not equal to 1
        probsB = [0.0, 1.0]
        answer = qu.HellingerDistance(probsA, probsB)
    except Exception as e:
        assert str(e) == "Both probability distributions must sum to 1."