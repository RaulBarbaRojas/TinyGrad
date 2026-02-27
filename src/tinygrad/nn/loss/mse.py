"""Implements the Mean Squared Error loss function as a Module.
"""

from typing import Literal

from tinygrad.core import Parameter
from tinygrad.nn.module import Module


class MSE(Module):
    """The MSE loss function implemented as a tinygrad's Module.
    """

    def __init__(self, agg_fn: Literal['sum', 'mean'] = 'mean') -> None:
        """Creates a new MSE loss function instance.

        :param agg_fn: The way the individual losses should be
        aggregated, defaults to `'mean'`.
        """
        super().__init__()
        self.agg_fn = agg_fn

    def forward(self, y_pred: list[Parameter],
                y_true: list[Parameter]) -> Parameter:
        """Runs MSE loss function.

        :param y_pred: The predicted values for N inputs.
        :param y_true: The ground truth values for N inputs.
        :return: The MSE loss function.
        """
        losses = [(pred_val - true_val) ** 2
                  for pred_val, true_val in zip(y_pred, y_true)]

        if self.agg_fn == 'sum':
            loss = sum(losses)
        elif self.agg_fn == 'mean':
            loss = sum(losses) / len(y_pred)
        else:
            raise ValueError(f"Unknown agg function '{self.agg_fn}'")

        return loss
