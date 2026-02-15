"""Implements a parameter with automatic gradient tracking.
"""

import math
from collections.abc import Callable
from enum import Enum
from typing import Self


class Operation(str, Enum):
    """An enumeration that showcases the operations that can lead to the
    creation of new parameters (used for gradient tracking and calculation).
    """

    MANUAL_CREATION = ''
    SUM: str = '+'
    MUL: str = '*'
    TANH: str = 'tanh'
    RELU: str = 'relu'


class Parameter:
    """A parameter with automatic gradient tracking.
    """

    def __init__(self, value: int | float,
                 _prev: tuple[Self] | None = None,
                 _op: str | Operation = Operation.MANUAL_CREATION) -> None:
        """Creates a new `Parameter` instance.

        :param value: The value of the parameter.
        will be extracted from it (ignoring user-provided args).
        :param _prev: The previous parameters that led to the creation
        of the parameter (if any), defaults to None
        :param _op: The operation that led to the creation of the
        parameter (if any), defaults to `Operation.MANUAL_CREATION`
        """
        self.value = value
        self._prev = set(_prev if _prev is not None else tuple())
        self._op = _op

        self.grad = 0.0
        self._backward: Callable = lambda: None

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
        param = Parameter(param) if not isinstance(param, Parameter) else param
        out_param = Parameter(self.value + param.value,
                              _prev=(self, param), _op=Operation.SUM)

        def _backward():
            self.grad += out_param.grad
            param.grad += out_param.grad

        out_param._backward = _backward

        return out_param

    def __mul__(self, param: int | float | Self) -> Self:
        """Multiplies a given parameter by the given object.

        :param param: The parameter to be multiplied with self.
        :return: A new parameter with the product result.
        """
        param = Parameter(param) if not isinstance(param, Parameter) else param
        out_param = Parameter(self.value * param.value,
                              _prev=(self, param), _op=Operation.MUL)

        def _backward():
            self.grad += param.value * out_param.grad
            param.grad += self.value * out_param.grad

        out_param._backward = _backward

        return out_param

    def tanh(self) -> Self:
        """Applies the tanh operation on the given parameter, creating a new
        one as a result.

        :return: A parameter obtained after applying tanh to the object.
        """
        exp_value = math.e ** (2 * self.value)
        out_param = Parameter((exp_value - 1) / (exp_value + 1),
                              _prev=(self, ), _op=Operation.TANH)

        def _backward():
            self.grad += (1.0 - out_param.value ** 2) * out_param.grad

        out_param._backward = _backward

        return out_param

    def relu(self) -> Self:
        """Applies the ReLU operation on the given parameter, creating a
        new one as a result.

        :return: A new parameter with the value obtained after
        applying the ReLU operation.
        """
        out_param = Parameter(0 if self.value < 0 else self.value,
                              _prev=(self, ), _op=Operation.RELU)

        def _backward() -> None:
            self.grad += (out_param.value > 0) * out_param.grad

        out_param._backward = _backward

        return out_param

    def backward(self) -> None:
        """Runs backpropagation, storing gradient values
        in all the parameters that affect the current object.

        NOTE: The backward propagation will be performed so that a
        node's backpropagation function gets called after the required
        backpropagations already occurred.
        """
        self.grad = 1.0

        topological_graph: list[Parameter] = []
        visited = set()

        def build_topological_graph(param: Parameter):
            if param in visited:
                return

            visited.add(param)
            for prev_param in param._prev:
                build_topological_graph(prev_param)

            topological_graph.append(param)

        build_topological_graph(self)
        for param in reversed(topological_graph):
            param._backward()
