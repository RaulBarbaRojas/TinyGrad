"""Implements a parameter with automatic gradient tracking.
"""

from typing import Self


class Parameter:
    """A parameter with automatic gradient tracking.
    """

    def __init__(self, value: int | float) -> None:
        """Creates a new `Parameter` instance.

        :param value: The value of the parameter.
        """
        self.value = value

    def __add__(self, param: int | float | Self) -> Self:
        """Adds a given parameter to the current object.

        :param param: The parameter to be added.
        :return: A new parameter with the value of the sum.
        """
        param = param if isinstance(param, Parameter) else Parameter(param)
        return Parameter(self.value + param.value)

    def __repr__(self) -> str:
        """Provides a string representation of the parameter object.

        :return: The parameter object as a string representation.
        """
        return f'Parameter({self.value})'
