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
        self.input_shape = x.shape
        if len(x.shape) <= 2:
            x.reshape([1] + list(x.shape))

        x_split = np.empty(shape=list(x.shape[:-2]) + [v // self.kernel_size for v in x.shape[-2:]] + [self.kernel_size, self.kernel_size])

        for i, ii in enumerate(range(0, x.shape[0]-self.kernel_size+1, self.stride)):
            for j, jj in enumerate(range(0, x.shape[1]-self.kernel_size+1, self.stride)):
                x_split[:, i, j] = x[:, ii:ii+self.kernel_size, jj:jj+self.kernel_size]

        return np.max(x_split, axis=(-2, -1))

    def backward(self, e):
        dims = len(e.shape)
        for i in range(2):
            temp_shape, new_shape = list(e.shape), list(e.shape)
            temp_shape.insert(dims-i, 1)
            new_shape[dims-i-1] *= self.kernel_size

            e = np.stack([e.reshape(temp_shape)] * self.kernel_size, axis=dims-i).reshape(new_shape)

        output = np.zeros(self.input_shape)
        output[:, :e.shape[-2], :e.shape[-1]] = e

        return output


class Convolution(Module):
    """
    2d convolutional layer.

    Derivation: https://compsci682-sp18.github.io/docs/conv2d_discuss.pdf
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

    def backward(self, delta):
        error = np.empty([1, *self.inputs.shape[-2:]])
        for i in range(self.out_channels):
            error[0] += convolve(delta[i], self.weight[i, 0, ::-1, ::-1], mode='full')
        error = np.repeat(error, self.in_channels, axis=0)

        delta = delta.reshape([self.out_channels, 1] + list(delta.shape[1:]))
        new_delta = np.zeros(self.weight.shape)
        for y_offset in range(self.kernel_size):
            for x_offset in range(self.kernel_size):
                xx = self.inputs.shape[-1] - self.kernel_size + x_offset + 1
                yy = self.inputs.shape[-2] - self.kernel_size + y_offset + 1
                input_partition = self.inputs[:, y_offset:yy, x_offset:xx] if len(self.inputs.shape) == 3 else self.inputs[y_offset:yy, x_offset:xx]
                self.weight[:, 0, y_offset, x_offset] -= np.sum(delta * input_partition, axis=tuple(range(len(delta.shape)))[1:])

        return error


__ALL__ = [Linear, MaxPool, Convolution]
