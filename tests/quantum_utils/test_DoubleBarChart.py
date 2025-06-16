from quantum_utils import QuantumUtils as qu


# Good cases
def test_DoubleBarChartValidProbabilites():
    exceptionOccured = ""
    try:
        qu.DoubleBarChart("dataA", [0.5, 0.5], "dataB", [0.5, 0.5], "SomeTitle")
    except Exception as e:
        exceptionOccured = str(e)
    finally:
        assert exceptionOccured == ""


# Bad cases
def test_DoubleBarChartNoTitleA():
    try:
        qu.DoubleBarChart("", [0.5, 0.5], "dataB", [0.5, 0.5], "SomeTitle")
    except Exception as e:
        assert str(e) == "Invalid parameter: lbl_A; empty string"


def test_DoubleBarChartNoTitleB():
    try:
        qu.DoubleBarChart("dataA", [0.5, 0.5], "", [0.5, 0.5], "SomeTitle")
    except Exception as e:
        assert str(e) == "Invalid parameter: lbl_B; empty string"


def test_DoubleBarChartNoDataA():
    try:
        qu.DoubleBarChart("dataA", [], "dataB", [0.5, 0.5], "SomeTitle")
    except Exception as e:
        assert str(e) == "Invalid parameter: data_A; empty list"


def test_DoubleBarChartNoDataB():
    try:
        qu.DoubleBarChart("dataA", [0.5, 0.5], "dataB", [], "SomeTitle")
    except Exception as e:
        assert str(e) == "Invalid parameter: data_B; empty list"


def test_BarChartInvalidDataLength():
    try:
        qu.DoubleBarChart("dataA", [0.5], "dataB", [0.5, 0.5], "SomeTitle")
    except Exception as e:
        assert str(e) == "Parameter invalid: numQubits; value must be smaller than 20 and greater than 0"
