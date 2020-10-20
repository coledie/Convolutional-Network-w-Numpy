"""
Loss functions.
"""
import numpy as np


class MeanSquaredError:
    """
    Mean squared error -- for regression problems.
    """
    def __call__(self, real, target):
        return (1 / len(real)) * np.sum((target - real)**2)

    def derivative(self, real, target):
        return 2 / len(real) * (real - target)


class CrossEntropyLoss:
    """
    Cross Entropy loss -- for categorical problems.
    """
    def __call__(self, real, target):
        return -real[np.where(target)] + np.log(np.sum(np.exp(real)))

    def derivative(self, real, target):
        return (1 / len(real)) * (real - target)


__ALL__ = [MeanSquaredError, CrossEntropyLoss]