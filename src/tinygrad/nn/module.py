"""Implements the most basic structure to define tinygrad nnets.
"""

from abc import ABC, abstractmethod
from typing import Any

from tinygrad.core import Parameter


class Module(ABC):
    """Defines a nnet in tinygrad.
    """

    def __init__(self) -> None:
        """Creates a tinygrad neural network instance.
        """
        self._params: dict[str, frozenset[Parameter]] = {}

    def parameters(self) -> frozenset[Parameter]:
        """Retrieves the parameters of the defined neural network.
        """
        return frozenset([param
                         for module_params in self._params.values()
                         for param in module_params])

    def zero_grad(self) -> None:
        """Zeroes-out the gradients of all the module's params.
        """
        for param in self.parameters():
            param.grad = 0

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets a given value as an attribute of the module.

        :param name: The name of the attribute.
        :param value: The value of the attribute.
        """
        if isinstance(value, Module):
            self._params[name] = value.parameters()
        elif isinstance(value, list):
            list_params: list[Parameter] = []
            for item in value:
                if isinstance(item, Parameter):
                    list_params.append(item)
                elif isinstance(item, Module):
                    list_params.extend(list(item.parameters()))

            if len(list_params) > 0:
                self._params[name] = frozenset(list_params)
        elif isinstance(value, Parameter):
            self._params[name] = frozenset([value])

        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Removes a given attribute from the module, and its parameters
        if any.

        :param name: The name of the attribute to be removed.
        """
        if name in self._params:
            del self._params[name]

        super().__delattr__(name)

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Runs the forward pass of the nnet.

        :return: The output of the forward pass of the nnet.
        """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
