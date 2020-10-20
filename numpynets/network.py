"""
Network implementations.
"""


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
        error = self.loss_prime(real, target) * self.learning_rate
        
        for module in self.modules[::-1]:
            error = module.backward(error)


__ALL__ = [Sequential]
