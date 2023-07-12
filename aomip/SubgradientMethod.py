import numpy as np
from .ForwardDifferences import FirstDerivative


class SGM:
    def __init__(
        self,
        A=None,
        b=None,
        x0=None,
        l=1e-3,
        beta=1,
        nmax=1000,
        eps=1e-6,
        constant=True,
        verbose=False,
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.l = l
        self.beta = beta
        self.nmax = nmax
        self.eps = eps
        self.constant = constant
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
            if not self.constant:
                l = 10*self.l / (i + 1)
            if self.verbose:
                x_vec.append(x0.reshape(shape))
                l_vec.append(l)
            ## update rule
            x = x0 - l * df(x0)
            x0 = np.maximum(x0, 0)
            ##
            err = np.linalg.norm(x - x0)
            x0 = x
            i = i + 1
        print("Number of iterations: {}".format(i))
        if self.verbose:
            return x.reshape(shape), x_vec, l_vec
        else:
            return x.reshape(shape)

    def l1Norm(self, beta=None):
        if beta is not None:
            self.beta = beta
        df = (
            lambda x: self.A.applyAdjoint(self.A.apply(x) - self.b).flatten()
            + self.beta * np.sign(FirstDerivative().apply(x)).flatten()
        )
        return self.gradDesc(df)
