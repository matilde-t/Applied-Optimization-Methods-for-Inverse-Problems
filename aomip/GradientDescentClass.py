import numpy as np
import scipy as sp


class GD:
    def __init__(self, A, b, x0, l=1e-3, nmax=1000, eps=1e-6):
        self.A = A
        self.b = b
        self.x0 = x0
        self.l = l
        self.nmax = nmax
        self.eps = eps

    def gradDesc(self, df):
        """
        Gradient descent.
        """
        print("Starting gradient descent")
        i = 0
        err = np.inf
        x0 = self.x0.copy()
        shape = x0.shape
        x0 = x0.flatten()
        while i < self.nmax and err > self.eps:
            x = x0 - self.l * df(x0)
            err = np.linalg.norm(x - x0)
            x0 = x
            i = i + 1
        print("Number of iterations: {}".format(i))
        return x.reshape(shape)

    def leastSquares(self):
        """
        Solve the Least Squares Problem using Gradient Descent.
        """
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
        return self.gradDesc(df)

    def l2Norm(self, L=None, beta=1):
        """
        Solves the Tikhonov problem in the form (1/2)||Ax-b||^2 + (beta/2)l^2||Lx||^2
        using gradient descent.
        """
        if L is None:
            L = sp.sparse.eye(len(self.x0.flatten()))
        df = (
            lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
            + beta / 2 * L.T @ L @ x.flatten()
        )
        return self.gradDesc(df)

    def huber(self, delta=1, beta=1):
        """
        Solves the Tikhonov problem with l1 regularization.
        """
        df = lambda x: self.A.applyAdjoint(
            self.A.apply(x) - self.b
        ) + beta / 2 * self.Ld(x, delta)
        return self.gradDesc(df)

    def fair(self, delta=1, beta=1):
        """
        Solves the Tikhonov problem with Fair potential.
        """
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b) + beta / 2 * x / (
            1 + x / delta
        )
        return self.gradDesc(df)


def forwardDiff(x):
    """
    Construct forward difference operator in 2D
    """
    length = np.prod(x.shape)
    n = x.shape[0]
    dx = sp.sparse.diags([-1, 1], [0, 1], (length, length))
    dy = sp.sparse.diags([-1, 1], [0, n + 1], (length, length))
    return sp.sparse.vstack([dx, dy])


def Ld(x, delta=1):
    """
    Derivative of the Huber function.
    """
    res = []
    shape = x.shape
    for i in x.flatten():
        if np.abs(i) < delta:
            res.append(i)
        else:
            res.append(delta * np.sign(i))
    return np.array(res).reshape(shape)
