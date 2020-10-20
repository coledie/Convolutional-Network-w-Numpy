"""
Convolutional neural network with Numpy.

MNIST Model from:
https://www.kaggle.com/andradaolteanu/convolutional-neural-nets-cnns-explained
"""
import numpy as np

from numpynets import *


def onehot(value, n_class):
    output = np.zeros(n_class)

    output[value] = 1.

    return output


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
        Reshape((28, 28)),
        Convolution(1, 16, 3),
        ReLu(),
        MaxPool(2),
        Convolution(16, 10, 3),
        ReLu(),
        MaxPool(2),
        Reshape((-1)),
        Linear(250, 128),
        ReLu(),
        Linear(128, 64),
        ReLu(),
        Linear(64, N_CLASS),
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
