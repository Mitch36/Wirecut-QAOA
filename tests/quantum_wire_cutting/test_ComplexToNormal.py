from quantum_wire_cutting import QuantumWireCutUtils as qwc

def test_ComplexToNormal():
    # Test with complex numbers
    probsA = [1j + 2j]
    answer = qwc.ComplexToNormal(probsA)
    assert answer == [3]  # Adjust this based on the expected output

def test_ComplexToNormalOneDecimal():
    # Test with complex numbers
    probsA = [0.1j + 0.2j]
    answer = qwc.ComplexToNormal(probsA)
    assert answer == [0.30000000000000004]  # Adjust this based on the expected output