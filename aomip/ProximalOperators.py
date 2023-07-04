import numpy as np


class ProximalOperators:
    def __init__(self, sigma=1e-3, y=0, prox_g=None, delta=0.1, l=1):
        self.sigma = sigma
        self.y = y
        self.prox_g = prox_g
        self.delta = delta
        self.l = l

    def constant(self, x, sigma=None):
        """
        Constant function (identity)
        """
        return x

    def translation(self, x, sigma=None):
        """
        Proximal operator of f(x) = g(x - y)
        """
        return self.y + self.prox_g(x - self.y, sigma)

    def l2(self, x, sigma=None):
        """
        Proximal operator of f(x) = l/2 * ||x||^2
        """
        if sigma is not None:
            self.sigma = sigma
        return x / (1 + self.sigma * self.l)

    def huber(self, x, sigma=None):
        """
        Proximal operator of f(x) = x^2/(2*delta) if |x| <= delta, else |x|
        """
        if sigma is not None:
            self.sigma = sigma
        return (1 - self.sigma / (np.maximum(np.abs(x), self.sigma) + self.delta)) * x

    def l21(self, x, sigma=None):
        """
        Proximal operator of f(x) = l * ||x||_21
        """
        if sigma is not None:
            self.sigma = sigma
        return (1 - self.l * self.sigma / np.maximum(np.linalg.norm(x), self.l)) * x

    def l11(self, x, sigma=None):
        """
        Proximal operator of f(x) = l * ||x||_11
        """
        if sigma is not None:
            self.sigma = sigma
        return self.soft(x, self.sigma * self.l)

    def soft(self, v, r):
        """
        Soft thresholding operator.
        """
        return np.sign(v) * np.maximum(np.abs(v) - r, 0)
