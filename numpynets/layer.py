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

        output_shape = [value // self.stride for value in x.shape[-2:]]

        self.max_locs = ([], [], [])  # c, y, x
        for channel in range(x.shape[0]):
            for i, ii in enumerate(range(0, x.shape[1]-self.kernel_size+1, self.stride)):
                for j, jj in enumerate(range(0, x.shape[2]-self.kernel_size+1, self.stride)):
                    region = x[channel, ii:ii+self.kernel_size, jj:jj+self.kernel_size]
                    region = region.flatten()

                    max_loc = np.argmax(region)

                    max_x_loc = max_loc % self.kernel_size
                    max_y_loc = max_loc // self.kernel_size

                    self.max_locs[0].append(channel)
                    self.max_locs[1].append(ii + max_y_loc)
                    self.max_locs[2].append(jj + max_x_loc)

        return x[self.max_locs].reshape([x.shape[0]] + output_shape)

    def backward(self, e):
        # output = np.zeros(self.input_shape)
        # output[self.max_locs] = e  # TODO shape of max_locs / may not give desired
        '''
        dims = len(e.shape)
        for i in range(2):
            temp_shape, new_shape = list(e.shape), list(e.shape)
            temp_shape.insert(dims-i, 1)
            new_shape[dims-i-1] *= self.kernel_size

            e = np.stack([e.reshape(temp_shape)] * self.kernel_size, axis=dims-i).reshape(new_shape)
        '''

        output = np.zeros(self.input_shape)
        output[self.max_locs] = e.flatten()

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
        self.bias = np.random.uniform(-.1, .1, size=self.out_channels)

    def forward(self, x):
        self.inputs = x
        if len(x.shape) <= 2:
            x = x.reshape([1, *x.shape])

        output = np.empty([self.out_channels] + [v-self.kernel_size+1 for v in x.shape[1:]])
        for i in range(self.out_channels):
            output[i] = np.sum(convolve(x, self.weight[i], mode='valid'), axis=0) + self.bias[i]

        return output

    def backward(self, delta):
        error = np.zeros([1, *self.inputs.shape[-2:]])
        for i in range(self.out_channels):
            error[0] += convolve(delta[i], self.weight[i, 0, ::-1, ::-1], mode='full')
        error = np.repeat(error, self.in_channels, axis=0)

        delta = delta.reshape([self.out_channels, 1] + list(delta.shape[1:]))
        for y_offset in range(self.kernel_size):
            for x_offset in range(self.kernel_size):
                xx = self.inputs.shape[-1] - self.kernel_size + x_offset + 1
                yy = self.inputs.shape[-2] - self.kernel_size + y_offset + 1
                input_partition = self.inputs[:, y_offset:yy, x_offset:xx] if len(self.inputs.shape) == 3 else self.inputs[y_offset:yy, x_offset:xx]
                self.weight[:, 0, y_offset, x_offset] -= np.sum(delta * input_partition, axis=tuple(range(len(delta.shape)))[1:])

        self.bias -= np.mean(delta, axis=tuple(range(len(delta.shape)))[1:])

        return error


__ALL__ = [Linear, MaxPool, Convolution]
