"""Implements an abstract definition of a gradlite module optimizer.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from gradlite.core.parameter import Parameter


class Optimizer(ABC):
    """Provides a template of what a gradlite optimizer looks like.
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 1e-4) -> None:
        """Creates a new optimizer.

        :param params: The parameters to be optimized.
        :param lr: The learning rate, defaults to 1e-4
        """
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        """Zero's out the gradients of the parameters.
        """
        for param in self.params:
            param.grad = 0

    @abstractmethod
    def step(self) -> None:
        """Updates the module's parameters.
        """
