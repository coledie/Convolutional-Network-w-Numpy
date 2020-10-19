"""
Layer implementations.
"""
import numpy as np

from module import Module


class Linear(Module):
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


class MaxPool(Module):
    """
    Max pool layer for 2d inputs.

    Parameters
    ----------
    size: int
        Size of kernel.
    stride: int, default=size
        How far kernel moves each step.
    padding: int
    """
    def __init__(self, size, stride=None):
        super().__init__()
        
        self.size = size
        self.stride = stride or size

    def forward(self, x):
        # Split up into blocks based on kernel size and stride
        s = self.stride
        x_split = []
        for i in range(x.shape[0] // self.stride):
            x_split.append([])
            for j in range(x.shape[1] // self.stride):
                x_split[-1].append(x[i*s:(i+1)*s, j*s:(j+1)*s])

        output = np.max(x_split, axis=-1)
        return output

    def backward(self, error):
        # Assumes x is one dimensional
        error = error.reshape((-1, 1))
        return np.stack([error] * self.size, axis=-1).flatten()


__ALL__ = [Linear, MaxPool]
