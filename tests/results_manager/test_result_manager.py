from graph import Graph
from results_manager import ResultsManager
from experiment_result import ExperimentResult
from experiment_configuration import ExperimentConfiguration as ExpConf

def test_ResultsManagerSingleMeasurement():
    # Create a sample graph
    graph = Graph([1, 1, 1])

    # Create a sample configuration
    configDict = {
        "experimentName": "test_experiment",
        "numQaoaLayers": 1,
        "shotsBudgetPerQuantumCircuit": 1000,
        "numSamples": 5,
        "maxClassicalOptimizationIterations": 30,
    }
    config = ExpConf(configDict)

    exampleDict = {
        'Optimized Parameters: ': 
        {'gamma': [1.16859592], 'beta': [1.16859592], 
        'costHistory': [(-0.616), (-1.109), (-1.168), (-1.08), (-1.566), (-0.635), (-1.586), (-1.542), (-1.121), (-1.625), (-1.642), (-1.658), (-1.609), (-1.666), (-1.628), (-1.65), (-1.646), (-1.627), (-1.627), (-1.683), (-1.609), (-1.64), (-1.637), (-1.635), (-1.652), (-1.649), (-1.64), (-1.647), (-1.63)]}, 
        'Probabilities': [0.016, 0.071, 0.349, 0.086, 0.081, 0.308, 0.069, 0.02 ], 
        'Counts': {'000': 0, '001': 0, '010': 500, '011': 0, '100': 0, '101': 0, '110': 0, '111': 500}
    }

    result = ExperimentResult(graph, config, exampleDict)

    # Create result manager
    manager: ResultsManager = ResultsManager()
    manager.AddResults(config.config["experimentName"], [result])

    exp = manager.Get("test_experiment")

    assert 0.5 == exp.GetApproximationRatio()
    assert 1.0 == exp.GetSuccesProbability()

def test_ResultsManagerDoubleMeasurements():
    # Create a sample graph
    graph = Graph([1, 1, 1])

    # Create a sample configuration
    configDict = {
        "experimentName": "test_experiment",
        "numQaoaLayers": 1,
        "shotsBudgetPerQuantumCircuit": 1000,
        "numSamples": 5,
        "maxClassicalOptimizationIterations": 30,
    }
    config = ExpConf(configDict)

    exampleDict = {
        'Optimized Parameters: ': 
        {'gamma': [1.16859592], 'beta': [1.16859592], 
        'costHistory': [(-0.616), (-1.109), (-1.168), (-1.08), (-1.566), (-0.635), (-1.586), (-1.542), (-1.121), (-1.625), (-1.642), (-1.658), (-1.609), (-1.666), (-1.628), (-1.65), (-1.646), (-1.627), (-1.627), (-1.683), (-1.609), (-1.64), (-1.637), (-1.635), (-1.652), (-1.649), (-1.64), (-1.647), (-1.63)]}, 
        'Probabilities': [0.016, 0.071, 0.349, 0.086, 0.081, 0.308, 0.069, 0.02 ], 
        'Counts': {'000': 0, '001': 0, '010': 500, '011': 0, '100': 0, '101': 0, '110': 0, '111': 500}
    }

    resultA = ExperimentResult(graph, config, exampleDict)
    resultB = ExperimentResult(graph, config, exampleDict)

    # Create result manager
    manager: ResultsManager = ResultsManager()
    manager.AddResults(config.config["experimentName"], [resultA, resultB])

    exp = manager.Get("test_experiment")

    assert 0.5 == exp.GetApproximationRatio()
    assert 1.0 == exp.GetSuccesProbability()

def test_ResultsManagerZeroSuccessProbability():
    # Create a sample graph
    graph = Graph([1, 1, 1])

    # Create a sample configuration
    configDict = {
        "experimentName": "test_experiment",
        "numQaoaLayers": 1,
        "shotsBudgetPerQuantumCircuit": 1000,
        "numSamples": 5,
        "maxClassicalOptimizationIterations": 30,
    }
    config = ExpConf(configDict)

    exampleDict = {
        'Optimized Parameters: ': 
        {'gamma': [1.16859592], 'beta': [1.16859592], 
        'costHistory': [(-0.616), (-1.109), (-1.168), (-1.08), (-1.566), (-0.635), (-1.586), (-1.542), (-1.121), (-1.625), (-1.642), (-1.658), (-1.609), (-1.666), (-1.628), (-1.65), (-1.646), (-1.627), (-1.627), (-1.683), (-1.609), (-1.64), (-1.637), (-1.635), (-1.652), (-1.649), (-1.64), (-1.647), (-1.63)]}, 
        'Probabilities': [0.016, 0.071, 0.349, 0.086, 0.081, 0.308, 0.069, 0.02 ], 
        'Counts': {'000': 500, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 500}
    }

    resultA = ExperimentResult(graph, config, exampleDict)
    resultB = ExperimentResult(graph, config, exampleDict)

    # Create result manager
    manager: ResultsManager = ResultsManager()
    manager.AddResults(config.config["experimentName"], [resultA, resultB])

    exp = manager.Get("test_experiment")

    assert 0.0 == exp.GetApproximationRatio()
    assert 0.0 == exp.GetSuccesProbability()