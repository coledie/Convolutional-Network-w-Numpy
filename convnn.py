"""
Convolutional neural network with Numpy.

Modelled after the PyTorch implementation: 
"""
import numpy as np


def onehot(value, n_class):
    output = np.zeros(n_class)

    output[value] = 1.

    return output


def sigmoid(x):
    return np.tanh(x)

def sigmoid_prime(x):
    return 1.0 - x**2


LEAK = 0
def relu(x):
    x[x < 0] = LEAK * x
    return x

def relu_prime(x):
    return np.where(x > 0, 1, LEAK)


activation_map = {
    'sigmoid': {"f": sigmoid, "f '": sigmoid_prime},
    'relu': {"f": relu, "f '": relu_prime},
    }


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


class NN:
    def __init__(self, layers, layer_activations, loss_prime, learning_rate=.5):
        self.layers = layers
        self.layer_activations = layer_activations
        self.learning_rate = learning_rate
        self.loss_prime = loss_prime

        assert len(self.layer_activations) == len(self.layers) - 1, "Number activations incorrect."

        self.w = []
        for i, layer in enumerate(self.layers[:-1]):
            # w[in, out]
            matrix = np.random.uniform(-.1, .1, size=(layer, self.layers[i+1]))

            self.w.append(matrix)

    def forward(self, x):
        """
        Network estimate y given x.
        """
        fires = [np.copy(x)]

        for i in range(len(self.layers) - 1):
            x = np.matmul(fires[-1], self.w[i])

            fires.append(activation_map[self.layer_activations[i]]['f'](x))

        return fires[-1], fires

    def backward(self, real, target, fires):
        """
        Update weights according to directional derivative to minimize error.
        """
        ## Error for output layer
        error = self.loss_prime(fires[-1], target)
        
        delta = activation_map[self.layer_activations[-1]]["f '"](fires[-1]) * error

        deltas = [delta]

        ## Backpropogate error
        for i in range(len(self.layers) - 3, -1, -1):
            error = np.sum(deltas[0] * self.w[i+1], axis=1)

            delta = activation_map[self.layer_activations[i]]["f '"](fires[i+1]) * error

            deltas.insert(0, delta)

        for i in range(len(self.layers) - 2, -1, -1):
            self.w[i] -= self.learning_rate * deltas[i] * fires[i].reshape((-1, 1))


import pandas as pd
train_dataset = pd.read_csv("mnist_train.csv")
test_dataset = pd.read_csv("mnist_test.csv")

train_X, train_y = train_dataset[train_dataset.columns[1:]].values, train_dataset['label'].values
test_X, test_y = test_dataset[test_dataset.columns[1:]].values, test_dataset['label'].values

## Setup NN
N_CLASS = 10
EPOCH = 10

cost = CrossEntropyLoss()

network = NN([784, 32, N_CLASS], ['sigmoid', 'sigmoid'], cost.derivative, 10**-4)

## Train
error = 0
for e in range(EPOCH):
    # shuffle dataset between epoch
    idx = [i for i in range(len(train_y))]
    np.random.shuffle(idx)
    train_X = train_X[idx]
    train_y = train_y[idx]

    for i, expected in enumerate(train_y):
        real, fires = network.forward(train_X[i])

        target = onehot(expected, N_CLASS)
        network.backward(real, target, fires)

        error += cost(real, target)

    if not e % 1:
        print(error)
        error = 0

        
## Evaluate
n_correct = 0
for i, expected in enumerate(test_y):
    real, fires = network.forward(test_X[i])

    n_correct += np.argmax(real) == expected

print(f"Correct: {n_correct / test_y.size:.2f}")
