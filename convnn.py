"""
Convolutional neural network with Numpy.

Base numpy nn implementation: https://www.kaggle.com/coledie/neural-network-w-numpy
MNIST Model from:
https://www.kaggle.com/andradaolteanu/convolutional-neural-nets-cnns-explained#3.-Convolutional-Neural-Networks-%F0%9F%8F%95%F0%9F%8F%9E%F0%9F%9B%A4%F0%9F%8F%9C%F0%9F%8F%96%F0%9F%8F%9D%F0%9F%8F%94
"""
import numpy as np


def onehot(value, n_class):
    output = np.zeros(n_class)

    output[value] = 1.

    return output


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


class ReLu:
    def __init__(self, leak):
        super().__init__()

        self.leak = leak

    def forward(self, x):
        self.output = np.copy(x)
        self.output[self.output < 0] *= self.leak
        return self.output

    def backward(self, error):
        return (np.where(x > 0, 1, self.leak)) * error


class Sigmoid:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, error):
        return (1.0 - self.output**2) * error


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


class Sequential:
    def __init__(self, modules, loss_prime, learning_rate):
        self.modules = modules
        self.loss_prime = loss_prime
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Network estimate y given x.
        """
        for module in self.modules:
            x = module.forward(x)

        return x

    def backward(self, real, target):
        """
        Update weights according to directional derivative to minimize error.
        """
        error = self.loss_prime(real, target)
        
        for module in self.modules[::-1]:
            error = module.backward(error)


if __name__ == '__main__':
    np.random.seed(0)

    ## Load datasets
    import pandas as pd
    train_dataset = pd.read_csv("mnist_train.csv")
    test_dataset = pd.read_csv("mnist_test.csv")

    train_X, train_y = train_dataset[train_dataset.columns[1:]].values, train_dataset['label'].values
    test_X, test_y = test_dataset[test_dataset.columns[1:]].values, test_dataset['label'].values

    ## Setup NN
    N_CLASS = 10
    EPOCH = 10

    cost = CrossEntropyLoss()

    network = Sequential([
        Linear(784, 32),
        Sigmoid(),
        Linear(32, N_CLASS),
        Sigmoid(),
    ], cost.derivative, 10**-4)

    ## Train
    error = 0
    for e in range(EPOCH):
        # shuffle dataset between epoch
        idx = [i for i in range(len(train_y))]
        np.random.shuffle(idx)
        train_X = train_X[idx]
        train_y = train_y[idx]

        for i, expected in enumerate(train_y):
            real = network.forward(train_X[i])

            target = onehot(expected, N_CLASS)
            network.backward(real, target)

            error += cost(real, target)

        if not e % 1:
            print(error)
            error = 0
            
    ## Evaluate
    n_correct = 0
    for i, expected in enumerate(test_y):
        real = network.forward(test_X[i])

        n_correct += np.argmax(real) == expected
    
    print(f"Correct: {n_correct / test_y.size:.2f}")
