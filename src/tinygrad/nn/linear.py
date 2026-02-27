"""Implements a linear layer in tinygrad.
"""

from random import uniform

from tinygrad.core.parameter import Parameter
from tinygrad.nn.activations import Identity
from tinygrad.nn.module import Module


class Neuron(Module):
    """Implements the traditional concept of DL neuron in tinygrad.
    """

    def __init__(self, input_features: int,
                 activation_fn: Module | None = None) -> None:
        """Creates a neuron.

        :param input_features: The input features of the neuron.
        :param activation_fn: The activation function of the neuron. If not
        given, the identity function will be used. Defaults to `None`
        """
        super().__init__()

        self.weights = [Parameter(uniform(-1.0, 1.0))  # nosec
                        for _ in range(input_features)]
        self.bias = Parameter(0.0)

        activation_fn = (activation_fn if activation_fn is not None
                         else Identity())
        self.activation_fn = activation_fn

    def forward(self, tensor: list[Parameter]) -> Parameter:
        """Runs the forward method of a basic neuron.

        :param tensor: The input tensor of the neuron.
        :returns: The output tensor of the neuron.
        """
        out_value = sum([input_val * weight + self.bias
                         for input_val, weight in zip(tensor, self.weights)])
        out_value = self.activation_fn(out_value)
        return out_value


class Linear(Module):
    """Implements a linear layer in tinygrad.
    """

    def __init__(self, input_features: int, output_features: int,
                 activation_fn: Module | None = None) -> None:
        """Creates a linear layer in tinygrad.

        :param input_features: The number of input features of the layer.
        :param output_features: The number of output features of the layer.
        :param activation_fn: The activation function of the neuron. If not
        given, the identity function will be used. Defaults to `None`
        """
        super().__init__()

        activation_fn = (activation_fn if activation_fn is not None
                         else Identity())

        self.neurons = [Neuron(input_features, activation_fn=activation_fn)
                        for _ in range(output_features)]

    def forward(self, tensor: list[Parameter]) -> list[Parameter]:
        """Runs the forward method of the linear layer-

        :param tensor: The input tensor of the linear layer.
        :return: The output obtained after the linear layer.
        """
        return [neuron(tensor) for neuron in self.neurons]
