import numpy as np


class ProximalOperators:
    def __init__(self, sigma=1, y=0, prox_g=None, delta=0.1, l=1):
        self.sigma = sigma
        self.y = y
        self.prox_g = prox_g
        self.delta = delta
        self.l = l

    def constant(self, x):
        """
        Constant function (identity)
        """
        return x

    def translation(self, x):
        """
        Proximal operator of f(x) = g(x - y)
        """
        return self.y + self.prox_g(x - self.y)

    def l2(self, x):
        """
        Proximal operator of f(x) = l/2 * ||x||^2
        """
        return x / (1 + self.sigma * self.l)

    def huber(self, x):
        """
        Proximal operator of f(x) = x^2/(2*delta) if |x| <= delta, else |x|
        """
        return (1 - self.sigma / (np.maximum(np.abs(x), self.sigma) + self.delta)) * x
