import numpy as np
import scipy as sp


class GD:
    def __init__(
        self,
        A=None,
        b=None,
        x0=None,
        l=1e-3,
        nmax=1000,
        eps=1e-6,
        backtrack=False,
        BB1=False,
        BB2=False,
        verbose=False,
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.l = l
        self.nmax = nmax
        self.eps = eps
        self.backtrack = backtrack
        self.BB1 = BB1
        self.BB2 = BB2
        self.verbose = verbose

    def gradDesc(self, df, f=None, x0=None):
        """
        Gradient descent.
        """
        print("Starting gradient descent")
        i = 0
        err = np.inf
        if self.x0 is not None:
            x0 = self.x0.copy()
        shape = x0.shape
        x0 = x0.flatten()
        l = self.l
        if self.verbose:
            x_vec = []
            l_vec = []
        while i < self.nmax and err > self.eps:
            if self.verbose:
                x_vec.append(x0.reshape(shape))
                l_vec.append(l)
            if self.backtrack:
                l = self.backtracking(df, f, x0)
            ## update rule
            x = x0 - l * df(x0)
            ##
            if self.BB1:
                s = x - x0
                y = df(x) - df(x0)
                l = np.dot(s, y) / np.dot(y, y)
            if self.BB2:
                s = x - x0
                y = df(x) - df(x0)
                l = np.dot(s, s) / np.dot(s, y)
            err = np.linalg.norm(x - x0)
            x0 = x
            i = i + 1
        print("Number of iterations: {}".format(i))
        if self.verbose:
            return x.reshape(shape), x_vec, l_vec
        else:
            return x.reshape(shape)

    def leastSquares(self):
        """
        Solve the Least Squares Problem using Gradient Descent.
        """
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
        f = lambda x: 0.5 * np.linalg.norm(self.A.apply(x) - self.b) ** 2
        return self.gradDesc(df, f)

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

    def backtracking(self, df, f, x0, rho=0.5, c=0.5):
        p = df(x0)
        alpha = 1
        while f(x0 - alpha * p) > f(x0) - c * alpha * np.linalg.norm(p) ** 2:
            alpha = rho * alpha
        return alpha


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
