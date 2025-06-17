from quantum_utils.src.quantum_utils import QuantumUtils as qu


# Good cases
def ToBinaryToDecimal00():
    result = qu.ToDecimal("00")
    assert result == 0


def ToBinaryToDecimalPretified00():
    result = qu.ToDecimal("|00>")
    assert result == 0


def test_ToDecimal01():
    result = qu.ToDecimal("01")
    assert result == 1


def test_ToDecimal10():
    result = qu.ToDecimal("10")
    assert result == 2


def ToBinaryToDecimalPretified10():
    result = qu.ToDecimal("|10>")
    assert result == 2


def test_ToDecimal11():
    result = qu.ToDecimal("11")
    assert result == 3


def ToBinaryToDecimalPretified11():
    result = qu.ToDecimal("|11>")
    assert result == 3


def test_ToDecimal15():
    result = qu.ToDecimal("1111")
    assert result == 15


def test_ToDecimal10000000():
    result = qu.ToDecimal("10000000")
    assert result == 128


# Bad cases
def test_ToDecimalMixedLetterAndNumbeInputr():
    try:
        result = qu.ToDecimal("A1C0")
    except ValueError as e:
        assert str(e) == "Invalid character in binaryStr: A1C0 must be either 0 or 1"


def test_ToDecimalLetterInput():
    try:
        result = qu.ToDecimal("ABCD")
    except ValueError as e:
        assert str(e) == "Invalid character in binaryStr: ABCD must be either 0 or 1"
