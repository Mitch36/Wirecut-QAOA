from quantum_utils import QuantumUtils as qu


# Good cases
def test_ToBinary0():
    result = qu.ToBinary(2, 0)
    assert result == "00"


def test_ToBinary1():
    result = qu.ToBinary(2, 1)
    assert result == "01"


def test_ToBinary2():
    result = qu.ToBinary(2, 2)
    assert result == "10"


def test_ToBinary3():
    result = qu.ToBinary(2, 3)
    assert result == "11"


def test_ToBinary15():
    result = qu.ToBinary(4, 15)
    assert result == "1111"


def test_ToBinary128():
    result = qu.ToBinary(8, 128)
    assert result == "10000000"


# Bad cases
def test_ToBinaryNotSuffecientBits15():
    result = qu.ToBinary(3, 15)
    assert result == "1111"


def test_ToBinaryNotSuffecientBits128():
    result = qu.ToBinary(7, 128)
    assert result == "10000000"
