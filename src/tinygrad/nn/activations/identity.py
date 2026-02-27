"""Implements the identity activation function as a tinygrad.nn.Module.
"""

from tinygrad.core import Parameter
from tinygrad.nn.module import Module


class Identity(Module):
    """Implements the identity activation function as nn.Module.
    """

    def forward(
        self, value: Parameter | list[Parameter]
    ) -> Parameter | list[Parameter]:
        """Runs the Identity forward pass.

        :param value: A tensor of parameters or a parameter.
        :return: The given input
        """
        return value
