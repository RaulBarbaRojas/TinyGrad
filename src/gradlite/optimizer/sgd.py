"""Implements default SGD optimizer in gradlite.
"""

from gradlite.optimizer.base import Optimizer


class SGD(Optimizer):
    """Implements the SGD optimizer in gradlite.
    """

    def step(self) -> None:
        for param in self.params:
            param.value -= self.lr * param.grad
