"""
Layer implementations.
"""
import numpy as np
from scipy.signal import convolve

from numpynets.module import Module


class Linear(Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.weight = np.random.uniform(-.1, .1, size=(self.n_outputs, self.n_inputs))
        self.bias = np.random.uniform(-.1, .1, size=self.n_outputs)

    def forward(self, x):
        self.inputs = x
        return np.matmul(self.weight, x) + self.bias

    def backward(self, delta):
        error = np.matmul(delta, self.weight)
        self.weight -= delta.reshape((-1, 1)) * self.inputs
        self.bias -= np.mean(delta.reshape((-1, 1)) * self.inputs, axis=1)
        return error


__ALL__ = [Linear]
