"""
Activation functions.
"""
import numpy as np

from numpynets.module import Module


class ReLu(Module):
    def __init__(self, leak=0):
        super().__init__()

        self.leak = leak

    def forward(self, x):
        self.output = np.copy(x)
        self.output[self.output < 0] *= self.leak
        return self.output

    def backward(self, error):
        return (np.where(error > 0, 1, self.leak)) * error


class Sigmoid(Module):
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, error):
        return (1.0 - self.output**2) * error


__ALL__ = [ReLu, Sigmoid]
