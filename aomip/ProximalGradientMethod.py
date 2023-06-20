import numpy as np


class PGM:
    def __init__(
        self,
        A=None,
        b=None,
        x0=None,
        function=None,
        beta=1e-2,
        l=1e-3,
        nmax=1000,
        backtrack=False,
        BB1=False,
        BB2=False,
        verbose=False,
        fast1=False,
        fast2=False,
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.function = function
        self.beta = beta
        self.l = l
        self.nmax = nmax
        self.backtrack = backtrack
        self.BB1 = BB1
        self.BB2 = BB2
        self.verbose = verbose
        self.fast1 = fast1
        self.fast2 = fast2

    def PGM(self, df, f=None, x0=None):
        """
        PGM algorithm.
        """
        print("Starting PGM")
        i = 0
        if self.x0 is not None:
            x0 = self.x0.copy()
        shape = x0.shape
        x0 = x0.flatten()
        z0 = x0.copy()
        t0 = 1
        l = self.l
        if self.verbose:
            x_vec = []
            l_vec = []
        while i < self.nmax:
            if self.verbose:
                x_vec.append(x0.reshape(shape))
                l_vec.append(l)
            if self.backtrack:
                l = self.backtracking(df, f, x0)
            ## update rule
            if self.fast1 or self.fast2:
                if self.fast1:
                    alpha = (i - 1) / (i + 2)
                if self.fast2:
                    t = (1 + np.sqrt(1 + 4 * t0**2)) / 2
                    alpha = (t0 - 1) / t
                    t0 = t
                z = self.function(x0 - l * df(x0))
                x = z + alpha * (z - z0)
                z0 = z
            else:
                x = self.function(x0 - l * df(x0))
            ##
            if self.BB1:
                s = x - x0
                y = df(x) - df(x0)
                l = np.dot(s, y) / np.dot(y, y)
            if self.BB2:
                s = x - x0
                y = df(x) - df(x0)
                l = np.dot(s, s) / np.dot(s, y)
            x0 = x
            i += 1
        print("PGM finished after {} iterations".format(i))
        if self.verbose:
            return x0.reshape(shape), x_vec, l_vec
        else:
            return x0.reshape(shape)

    def leastSquares(self):
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
        f = lambda x: 0.5 * np.linalg.norm(self.A.apply(x) - self.b) ** 2
        return self.PGM(df, f)

    def backtracking(self, df, f, x0, rho=0.5, c=0.5):
        p = df(x0)
        alpha = 1
        while f(x0 - alpha * p) > f(x0) - c * alpha * np.linalg.norm(p) ** 2:
            alpha = rho * alpha
        return alpha
