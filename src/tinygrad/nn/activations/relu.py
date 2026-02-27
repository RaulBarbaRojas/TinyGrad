"""Implements the ReLU activation function as a tinygrad.nn.Module.
"""

from tinygrad.core import Parameter
from tinygrad.nn.module import Module


class ReLU(Module):
    """Implements the ReLU activation function as nn.Module.
    """

    def forward(
        self, value: Parameter | list[Parameter]
    ) -> Parameter | list[Parameter]:
        """Runs the ReLU forward pass.

        :param value: A tensor of parameters or a parameter.
        :return: A tensor of parameters or a parameter with ReLU applied
        """
        if isinstance(value, Parameter):
            return value.relu()

        return [param.relu() for param in value]
        return [param.relu() for param in value]
        return [param.relu() for param in value]
