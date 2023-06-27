import numpy as np
import scipy as sp


class POGM:
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
        nonneg=False,
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
        self.nonneg = nonneg

    def POGM(self, df, f=None, x0=None, function=None):
        """
        POGM algorithm.
        """
        print("Starting POGM")
        i = 0
        if function is not None:
            self.function = function
        if self.x0 is not None:
            x0 = self.x0.copy()
        shape = x0.shape
        ## initialization
        x0 = x0.flatten()
        z0 = x0.copy()
        omega0 = x0.copy()
        theta0 = 1
        gamma0 = 1
        l = 1 / self.l
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
            if i == self.nmax - 1:
                theta = 0.5 * (1 + np.sqrt(8 * theta0**2 + 1))
            else:
                theta = 0.5 * (1 + np.sqrt(4 * theta0**2 + 1))
            gamma = 1 / l * (2 * theta0 + theta - 1) / theta
            omega = x0 - 1 / l * df(x0)
            z = (
                omega
                + (theta0 - 1) / theta * (omega - omega0)  # Nesterov
                + theta0 / theta * (omega - x0)  # OGM
                + (theta0 - 1) / (l * gamma0 * theta) * (z0 - x0)  # POGM
            )
            x = self.function(z, gamma)
            if self.nonneg:
                x = np.maximum(x, 0)
            i += 1
            ##
            if self.BB1:
                s = x - x0
                y = df(x) - df(x0)
                l = 1 / (np.dot(s, y) / np.dot(y, y))
            if self.BB2:
                s = x - x0
                y = df(x) - df(x0)
                l = 1 / (np.dot(s, s) / np.dot(s, y))
            ## exchange values
            theta0 = theta
            gamma0 = gamma
            omega0 = omega
            z0 = z
            x0 = x

        print("POGM finished after {} iterations".format(i))
        if self.verbose:
            return x0.reshape(shape), x_vec, l_vec
        else:
            return x0.reshape(shape)

    def leastSquares(self):
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
        f = lambda x: 0.5 * np.linalg.norm(self.A.apply(x) - self.b) ** 2
        return self.POGM(df, f)

    def backtracking(self, df, f, x0, rho=0.5, c=0.5):
        p = df(x0)
        alpha = 1
        while f(x0 - alpha * p) > f(x0) - c * alpha * np.linalg.norm(p) ** 2:
            alpha = rho * alpha
        return alpha

    def l2Norm(self, L=None, beta=1):
        if L is None:
            L = sp.sparse.eye(len(self.x0.flatten()))
        df = (
            lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
            + beta / 2 * L.T @ L @ x.flatten()
        )
        return self.POGM(df)
