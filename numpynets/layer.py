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

    def backward(self, error):
        self.weight -= error.reshape((-1, 1)) * self.inputs
        self.bias -= np.mean(error.reshape((-1, 1)) * self.inputs, axis=1)
        return np.matmul(error, self.weight)


class MaxPool(Module):
    """
    Max pool layer for 2d inputs.

    Parameters
    ----------
    kernel_size: int
        Size of kernel.
    stride: int, default=size
        How far kernel moves each step.
    padding: int
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: ndarray
            Square input image.
        """
        if len(x.shape) <= 2:
            x.reshape([1] + list(x.shape))

        x_split = np.empty(shape=list(x.shape[:-2]) + [v // self.kernel_size for v in x.shape[-2:]] + [self.kernel_size, self.kernel_size])

        for i, ii in enumerate(range(0, x.shape[0]-self.kernel_size+1, self.stride)):
            for j, jj in enumerate(range(0, x.shape[1]-self.kernel_size+1, self.stride)):
                x_split[:, i, j] = x[:, ii:ii+self.kernel_size, jj:jj+self.kernel_size]

        return np.max(x_split, axis=(-2, -1))

    def backward(self, error):
        """
        Backward pass.

        Parameters
        ----------
        error: ndarray
            1d error.
        """
        error = error.reshape((-1, 1))
        return np.stack([error] * self.kernel_size**2, axis=-1).flatten()


class Convolution(Module):
    """
    2d convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = np.random.uniform(-.1, .1, size=(self.out_channels, 1, self.kernel_size, self.kernel_size))

    def forward(self, x):
        self.inputs = x
        if len(x.shape) <= 2:
            x = x.reshape([1, *x.shape])

        output = np.empty([self.out_channels] + [v - self.kernel_size+1 for v in x.shape[1:]])
        for i in range(self.out_channels):
            output[i] = np.sum(convolve(x, self.weight[i], mode='valid'), axis=0)

        return output

    def backward(self, error):
        self.weight -= error.reshape((-1, 1)) * self.inputs
        return np.matmul(error, self.weight)


__ALL__ = [Linear, MaxPool, Convolution]
