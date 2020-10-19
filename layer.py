"""
Layer implementations.
"""
import numpy as np

from module import Module


class Linear:
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.weight = np.random.uniform(-.1, .1, size=(self.n_outputs, self.n_inputs))

    def forward(self, x):
        self.inputs = x
        return np.matmul(self.weight, x)

    def backward(self, error):
        self.weight -= error.reshape((-1, 1)) * 10**-4 * self.inputs
        return np.matmul(error, self.weight)


__ALL__ = [Linear]
