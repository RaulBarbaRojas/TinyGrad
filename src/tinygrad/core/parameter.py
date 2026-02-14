"""Implements a parameter with automatic gradient tracking.
"""

from enum import Enum
from typing import Self


class Operation(str, Enum):
    """An enumeration that showcases the operations that can lead to the
    creation of new parameters (used for gradient tracking and calculation).
    """

    MANUAL_CREATION = ''
    SUM: str = '+'
    MUL: str = '*'


class Parameter:
    """A parameter with automatic gradient tracking.
    """

    def __init__(self, value: int | float | Self,
                 _prev: tuple[Self] | None = None,
                 _op: str | Operation = Operation.MANUAL_CREATION) -> None:
        """Creates a new `Parameter` instance.

        :param value: The value of the parameter (or a parameter).
        :param _prev: The previous parameters that led to the creation
        of the parameter (if any), defaults to None
        :param _op: The operation that led to the creation of the
        parameter (if any), defaults to `Operation.MANUAL_CREATION`
        """
        value = value.value if isinstance(value, Parameter) else value
        self.value = value
        self._prev = set(_prev if _prev is not None else tuple())
        self._op = _op

        self.grad = 0.0

    def __repr__(self) -> str:
        """Provides a string representation of the parameter object.

        :return: The parameter object as a string representation.
        """
        return f'Parameter({self.value})'

    def __add__(self, param: int | float | Self) -> Self:
        """Adds a given parameter to the current object.

        :param param: The parameter to be added.
        :return: A new parameter with the value of the sum.
        """
        out_param = Parameter(self.value + Parameter(param).value,
                              _prev=(self, param), _op=Operation.SUM)
        return out_param

    def __mul__(self, param: int | float | Self) -> Self:
        """Multiplies a given parameter by the given object.

        :param param: The parameter to be multiplied with self.
        :return: A new parameter with the product result.
        """
        out_param = Parameter(Parameter(param).value * self.value,
                              _prev=(self, param), _op=Operation.MUL)
        return out_param
