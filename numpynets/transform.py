"""
Array transforms.
"""
from numpynets.module import Module


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()

        self.shape = shape
    
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(self.shape)
    
    def backward(self, x):
        return x.reshape(self.original_shape)


__ALL__ = [Reshape]
