"""Implements a parameter with automatic gradient tracking.
"""

from typing import Self


class Parameter:
    """A parameter with automatic gradient tracking.
    """

    def __init__(self, value: int | float | Self) -> None:
        """Creates a new `Parameter` instance.

        :param value: The value of the parameter (or a parameter).
        """
        value = value.value if isinstance(value, Parameter) else value
        self.value = value

    def __add__(self, param: int | float | Self) -> Self:
        """Adds a given parameter to the current object.

        :param param: The parameter to be added.
        :return: A new parameter with the value of the sum.
        """
        return Parameter(self.value + Parameter(param).value)

    def __mul__(self, param: int | float | Self) -> Self:
        """Multiplies a given parameter by the given object.

        :param param: The parameter to be multiplied with self.
        :return: A new parameter with the product result.
        """
        return Parameter(Parameter(param).value * self.value)

    def __repr__(self) -> str:
        """Provides a string representation of the parameter object.

        :return: The parameter object as a string representation.
        """
        return f'Parameter({self.value})'
