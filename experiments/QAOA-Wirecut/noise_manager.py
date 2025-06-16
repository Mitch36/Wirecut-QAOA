from quantum_backend import QuantumBackEnd
from qiskit_ibm_runtime import QiskitRuntimeService
import pickle
import os
from qiskit_aer.noise import NoiseModel


class NoiseManager:

    @staticmethod
    def ReadNoiseModel(backend: QuantumBackEnd) -> NoiseModel | None:
        """
        Read the binary file containing the noise model for the specified backend

        Args:
            backend (QuantumBackEnd): The backend to read the noise model for; currently only Qiskit-Aer-IBM_Sherbrooke and Qiskit-Aer-IBM_Brisbane are supported).

        Returns:
            NoiseModel | None: The noise model for the specified backend, or None if the file does not exist.

        Raises:
            Exception: If the backend is not supported or the noise model file is not of type NoiseModel.
        """

        if not backend == QuantumBackEnd.QISKIT_AER_IBM_BRISBANE and not backend == QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
            raise Exception(f"Cannot read noise model for ideal or quantum hardware backends; received: {backend.value}")

        fileName = f"bin/{backend.value}-noisemodel.pkl"
        if os.path.exists(fileName):
            with open(fileName, 'rb') as file:
                noiseModel = pickle.load(file)
                if not isinstance(noiseModel, NoiseModel):
                    raise Exception(f"Noise model file is not of type NoiseModel; received: {type(noiseModel)}")
                return noiseModel
        return None

    @staticmethod
    def WriteNoiseModel(model: NoiseModel, backend: QuantumBackEnd) -> str:
        """
        Writes the NoiseModel object to a binary file in bin directory.

        Args:
            model (NoiseModel): The noise model to save.
            backend (QuantumBackEnd): The backend to save the noise model for.

        Returns:
            str: The file path of the saved noise model
        """

        if not backend == QuantumBackEnd.QISKIT_AER_IBM_BRISBANE and not backend == QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
            raise Exception(f"Cannot save noise model for ideal or quantum hardware backends; received: {backend.value}")

        fileName = f"bin/{backend.value}-noisemodel.pkl"
        with open(fileName, 'wb') as file:
            pickle.dump(model, file)
        return fileName

    @staticmethod
    def FetchNoiseModel(backend: QuantumBackEnd) -> NoiseModel:
        """
        Fetches the noise model from IBM Quantum Platform based on the backend provided.
        Args:
            backend (QuantumBackEnd): The quantum backend to fetch the noise model from.
        Returns:
            NoiseModel: The noise model for the specified backend.
        Raises:
            ValueError: If the backend is not supported or the IBM Quantum Platform is not available.
        """
        service = QiskitRuntimeService()
        ibmBackend = None
        match backend:
            case QuantumBackEnd.QISKIT_AER_IBM_SHERBROOKE:
                ibmBackend = service.backend(name="ibm_sherbrooke", instance="ibm-q/open/main")
            case QuantumBackEnd.QISKIT_AER_IBM_BRISBANE:
                ibmBackend = service.backend(name="ibm_brisbane", instance="ibm-q/open/main")
            case _: 
                raise ValueError(f"Unsupported backend provided; received: {backend.value}")
                
        return NoiseModel.from_backend(ibmBackend)