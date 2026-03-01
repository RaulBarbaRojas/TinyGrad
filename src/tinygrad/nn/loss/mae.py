"""Implements the Mean Absolute Error (MAE) loss function in tinygrad.
"""

from typing import Literal

from tinygrad.core.parameter import Parameter
from tinygrad.nn.module import Module


class MAE(Module):
    """An implementation of the Mean Absolute Error loss in tinygrad.
    """

    def __init__(self, agg_fn: Literal['sum', 'mean'] = 'mean') -> None:
        """Creates a new MAE loss function instance.

        :param agg_fn: The way the individual losses should be
        aggregated, defaults to `'mean'`.
        """
        super().__init__()
        self.agg_fn = agg_fn

    def forward(self, y_pred: list[Parameter],
                y_true: list[Parameter]) -> Parameter:
        losses = [(pred_param - true_param).abs()
                  for pred_param, true_param in zip(y_pred, y_true)]

        if self.agg_fn == 'sum':
            loss = sum(losses)
        elif self.agg_fn == 'mean':
            loss = sum(losses) / len(y_pred)
        else:
            raise ValueError(f"Unknown agg function '{self.agg_fn}'")

        return loss
