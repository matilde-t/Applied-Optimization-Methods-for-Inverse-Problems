import numpy as np


class ISTA:
    def __init__(
        self,
        A=None,
        b=None,
        x0=None,
        beta=1e-2,
        l=1e-3,
        nmax=1000,
        eps=1e-6,
        backtrack=False,
        BB1=False,
        BB2=False,
        debug=False,
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.beta = beta
        self.l = l
        self.nmax = nmax
        self.eps = eps
        self.backtrack = backtrack
        self.BB1 = BB1
        self.BB2 = BB2
        self.debug = debug

    def ISTA(self, df, f=None, x0=None):
        """
        ISTA algorithm.
        """
        print("Starting ISTA")
        i = 0
        err = np.inf
        if self.x0 is not None:
            x0 = self.x0.copy()
        shape = x0.shape
        x0 = x0.flatten()
        l = self.l
        if self.debug:
            x_vec = []
            l_vec = []
        while i < self.nmax and err > self.eps:
            if self.debug:
                x_vec.append(x0.reshape(shape))
                l_vec.append(l)
            if self.backtrack:
                l = self.backtracking(df, f, x0)
            ## update rule
            x = soft(x0 - l * df(x0), self.beta * l)
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
            i += 1
        print("ISTA finished after {} iterations".format(i))
        if self.debug:
            return x0.reshape(shape), x_vec, l_vec
        else:
            return x0.reshape(shape)

    def leastSquares(self):
        """
        Insert Least Squares into the LASSO problem.
        """
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
        f = lambda x: 0.5 * np.linalg.norm(self.A.apply(x) - self.b) ** 2
        return self.ISTA(df, f)

    def backtracking(self, df, f, x0, rho=0.5, c=0.5):
        p = df(x0)
        alpha = 1
        while f(x0 - alpha * p) > f(x0) - c * alpha * np.linalg.norm(p) ** 2:
            alpha = rho * alpha
        return alpha


def soft(v, r):
    return np.sign(v) * np.maximum(np.abs(v) - r, 0)
