import datetime
       

class ExperimentConfiguration:

    # Configuration keys that must be defined
    mustDefinedConfigKeys = {
        "experimentName": str,
        "numQaoaLayers": int, # If wire cutting is used, one layer is the maximum. Further research required for implementation of multiple layers.
        "shotsBudgetPerQuantumCircuit": int,
        "numSamples": int,
        "maxClassicalOptimizationIterations": int,
        "classicalOptimizationAlgorithm": str # Can be COBYLA or SPSA.
    }

    # Configuration keys that are static pre-defined
    constConfigKeys = {
        # Quantum circuit information
        "numOriginalQubits": "To be defined based on Graph object", # Number of qubits which are simulated
        "numQubits": "To be defined based on Graph object", # Number of physical qubits required to run the circuit 
        "numCuts": "To be defined based on Graph object",
        "quantumCircuitType": "QAOA-MAXCUT",
        "quantumCircuitBackend": "To be defined", # To be defined by run functions used in Circuit and WireCutCircuit functions.

        # Classical resource information
        "classicalOptimizationProvider": "To be defined by WireCutQAOA class",

        # QAOA Training data, most optimal parameters
        "gamma": "To be defined by optimization algorithm",
        "beta": "To be defined by optimization algorithm",
        "optimalAvgCost": "To be defined by optimization algorithm",
        "numClassicalOptimizationIterations": "To be defined by optimization algorithm",

        # Date and time of the experiment
        "dateTime": "YYYY-MM-DD HH:mm:ss", # ISO 8601 format
    }

    @staticmethod
    def __validate_config__(config: dict) -> None:
        """
        Validates the configuration dictionary if all 'must defined' keys are present and of the correct type.

        Args:
            config (dict): The configuration dictionary to validate.

        Raises:
            ValueError: If one or multiple must defined configuration keys are missing.
            ValueError: If the type of any must defined configuration key is incorrect.
        """
            
        for mustDefinedKey in ExperimentConfiguration.mustDefinedConfigKeys:
            if mustDefinedKey not in config:
                raise ValueError(f"Invalid configuration: {mustDefinedKey} key and value are missing; must defined keys are {ExperimentConfiguration.mustDefinedConfigKeys.keys()}")
                
        # All must defined keys are present, now check if they are of the correct type
        for key, value in config.items():
            # There is only one key which must be of type string
            # Handle string type variables 
            if key == "experimentName":
                if not isinstance(value, str):
                    raise ValueError(f"Invalid configuration: {key} must be of type string")
                elif len(value) < 1:
                    raise ValueError(f"Invalid configuration: {key} must be a non-empty string")
                elif len(value) > 100:
                    raise ValueError(f"Invalid configuration: {key} must be a string of length less than 100 characters; experimentName value is {value}")
            
            # Handle int type variables    
            if key == "numQubits" or key == "numCuts" or key == "numLayers" or key == "shotsBudgetPerQuantumCircuit" or key == "numSamples":
                if not isinstance(value, int):
                    raise ValueError(f"Invalid configuration: {key} must be of type int")
                elif value < 1:
                    raise ValueError(f"Invalid configuration: {key} must be a positive integer; value is {value}")

    def __init__(self, config: dict):
        """
        Constructor for the ExperimentConfiguration class.

        Args:
            config (dict): A dictionary containing the configuration parameters for the experiment.

        Raises:
            ValueError: If the configuration does not contain all must defined keys.
        """
        ExperimentConfiguration.__validate_config__(config) # Throws value error if not valid

        constConfig = ExperimentConfiguration.constConfigKeys
        constConfig["dateTime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config = config
        self.config.update(constConfig)

    def GetConfig(self) -> dict:
        """
        Get the configuration dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return self.config
    
    def Get(self, key: str) -> any:
        """
        Get the value of a specific configuration key, if key does not exist return None.

        Args:
            key (str): The configuration key to retrieve.

        Returns:
            any: The value associated with the specified key.
        """
        return self.config.get(key, None)
    
    def SetName(self, name: str) -> None:
        """
        Set the name of the experiment.

        Args:
            name (str): The name of the experiment.
        """
        self.config["experimentName"] = name
        
    def __str__(self) -> str:
        """
        String representation of the ExperimentConfiguration object.

        Returns:
            str: A string representation of the configuration.
        """
        outputStr = "ExperimentConfiguration: \n"
        for key, value in self.config.items():
            outputStr += f"  {key}: {value}\n"
        return outputStr