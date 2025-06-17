from quantum_utils.src.quantum_utils import QuantumUtils as qu


# Good cases
# On state 0000
def test_FlipBitValidInputX000():
    result = qu.FlipBit("0000", 0)
    assert result == "1000"


def test_FlipBitValidInput0X00():
    result = qu.FlipBit("0000", 1)
    assert result == "0100"


def test_FlipBitValidInput00X0():
    result = qu.FlipBit("0000", 2)
    assert result == "0010"


def test_FlipBitValidInput000X():
    result = qu.FlipBit("0000", 3)
    assert result == "0001"


# on state 1111
def test_FlipBitValidInputX111():
    result = qu.FlipBit("1111", 0)
    assert result == "0111"


def test_FlipBitValidInput1X11():
    result = qu.FlipBit("1111", 1)
    assert result == "1011"


def test_FlipBitValidInput11X1():
    result = qu.FlipBit("1111", 2)
    assert result == "1101"


def test_FlipBitValidInput111X():
    result = qu.FlipBit("1111", 3)
    assert result == "1110"


# negative index
def test_FlipBitValidInputX000NegativeIndex():
    result = qu.FlipBit("0000", -4)
    assert result == "1000"


def test_FlipBitValidInput0X00NegativeIndex():
    result = qu.FlipBit("0000", -3)
    assert result == "0100"


def test_FlipBitValidInput00X0NegativeIndex():
    result = qu.FlipBit("0000", -2)
    assert result == "0010"


def test_FlipBitValidInput000XNegativeIndex():
    result = qu.FlipBit("0000", -1)
    assert result == "0001"


# Bad cases
def test_FlipBitIndexOutOfBounds():
    try:
        qu.FlipBit("0000", 4)
    except Exception as e:
        assert str(e) == "Invalid parameter: index; out of bounds"


def test_FlipBitEmptyString():
    try:
        qu.FlipBit("", 2)
    except Exception as e:
        assert str(e) == "Invalid parameter: binaryStr; empty string"


def test_FlipBitInvalidChar():
    try:
        qu.FlipBit("ABCD", 0)
    except Exception as e:
        assert str(e) == "Invalid character in state: ABCD must be either 0 or 1"
