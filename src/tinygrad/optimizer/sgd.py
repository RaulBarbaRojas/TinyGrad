"""Implements default SGD optimizer in tinygrad.
"""

from tinygrad.optimizer.base import Optimizer


class SGD(Optimizer):
    """Implements the SGD optimizer in tinygrad.
    """

    def step(self) -> None:
        for param in self.params:
            param.value -= self.lr * param.grad
