"""
Base module template class.
"""
import numpy as np


class Module:
    """
    Differentiable piece of neural network.

    self.weight: ndarray
        Weights of module.
    self.bias: ndarray or None
        Bias of module.
    """
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        raise NotImplementedError(f"{type(self)}.forward not implemented!")

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        """
        raise NotImplementedError(f"{type(self)}.forward not backward!")
