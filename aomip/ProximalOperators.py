import numpy as np


class ProximalOperators:
    def __init__(self):
        pass

    def indicator(self, x, c, sigma=None):
        """
        Indicator function of a interval c
        """
        return 0 if (x >= np.min(c) and x <= np.max(c)) else np.inf

    def constant(self, x, sigma=None):
        """
        Constant function (identity)
        """
        return x

    def translation(self, x, prox_g, y, sigma=None):
        """
        Proximal operator of f(x) = g(x - y)
        """
        return y + prox_g(x - y, sigma)

    def l2(self, x, sigma, l=1):
        """
        Proximal operator of f(x) = l/2 * ||x||^2
        """
        return x / (1 + sigma * l)

    def huber(self, x, delta, sigma):
        """
        Proximal operator of f(x) = x^2/(2*delta) if |x| <= delta, else |x|
        """
        return (1 - sigma / (max(np.linalg.norm(x), sigma) + delta)) * x

    def l1(self, x, sigma):
        """
        L1 norm proximal operator (soft thresholding)
        """
        return np.sign(x) * np.maximum(np.abs(x) - sigma, 0)
