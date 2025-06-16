from circuit_arguments.circuit_argument_interface import CircuitArgumentInterface
from circuit_configurations.circuit_configuration_interface import (
    CircuitConfigurationInterface,
)


class CircuitConfiguration(CircuitConfigurationInterface):
    def __init__(self, args: list):
        self.args = args

    def Add(self, arg: CircuitArgumentInterface):
        self.args.append(arg)

    def Find(self, qubit: int) -> list[CircuitArgumentInterface]:
        """
        Finds all arguments related to a specific qubit.

        Args:
            qubit (int): The qubit to search for.

        Returns:
            list[CircuitArgument]: A list of arguments that affect the qubit.

        Raises:
            Exception: If the qubit is negative.
        """

        if qubit < 0:
            raise ValueError("Parameter invalid: qubit; Qubit cannot be negative")

        result = []
        for arg in self.args:
            if arg.qubit == qubit:
                result.append(arg)
        return result

    def __str__(self) -> str:
        string = "CircuitConfiguration: "
        for arg in self.args:
            string += str(arg) + "\n"
        return string
