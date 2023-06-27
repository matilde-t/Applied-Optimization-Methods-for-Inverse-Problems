import numpy as np
import scipy as sp


class OGM1:
    def __init__(self, A, b, x0, l=1e-3, nmax=1000, theta0=1, verbose=False, nonneg=False):
        self.A = A
        self.b = b
        self.x0 = x0
        self.l = l
        self.nmax = nmax
        self.theta0 = theta0
        self.verbose = verbose
        self.nonneg = nonneg

    def OGM1(self, df):
        """
        Optimized Gradient Method 1, based on the paper by Kim and Fessler.
        """
        print("Starting Optimized Gradient Method 1")
        shape = self.x0.shape
        x0 = self.x0.copy().flatten()
        theta0 = self.theta0
        y0 = x0
        if self.verbose:
            x_vec = []
            l_vec = []
        for i in range(0, self.nmax):
            if self.verbose:
                x_vec.append(x0.reshape(shape))
                l_vec.append(self.l)
            y = x0 - self.l * df(x0)
            theta = (1 + np.sqrt(1 + 4 * theta0**2)) / 2
            x = y + (theta0 - 1) / theta * (y - y0) + theta0 / theta * (y - x0)
            y0 = y
            theta0 = theta
            x0 = x
        theta = (1 + np.sqrt(1 + 8 * theta0**2)) / 2
        x = y + (theta0 - 1) / theta * (y - y0) + theta0 / theta * (y - x0)
        if self.nonneg:
            x0 = np.maximum(x0, 0)
        print("Finished Optimized Gradient Method 1")
        if self.verbose:
            return x0.reshape(shape), x_vec, l_vec
        else:
            return x0.reshape(shape)

    def leastSquares(self):
        """
        Solve the Least Squares Problem using OGM1.
        """
        df = lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
        return self.OGM1(df)

    def l2Norm(self, L=None, beta=1):
        """
        Solves the Tikhonov problem in the form (1/2)||Ax-b||^2 + (beta/2)l^2||Lx||^2
        using OGM1.
        """
        if L is None:
            L = sp.sparse.eye(len(self.x0.flatten()))
        df = (
            lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
            + beta / 2 * L.T @ L @ x.flatten()
        )
        return self.OGM1(df)
