import numpy as np
from .ProximalOperators import ProximalOperators as PO


class ADMM:
    def __init__(
        self,
        A=None,
        b=None,
        x0=None,
        prox_f=None,
        prox_g=None,
        mu=1,
        tau=1,
        nmax=1000,
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.mu = mu
        self.tau = tau
        self.nmax = nmax

    def ADMM(self):
        """
        ADMM algorithm.
        """
        print("Starting ADMM")

        i = 0
        shape = self.x0.shape
        x0 = self.x0.copy()
        z0 = self.A.apply(x0)
        u0 = np.zeros(self.b.shape)

        while i < self.nmax:
            x = self.prox_f(
                x0
                - self.mu / self.tau * self.A.applyAdjoint(self.A.apply(x0) - z0 + u0)
            )
            z = self.prox_g(self.A.apply(x) + u0)
            u = u0 + self.A.apply(x) - z
            x0 = x
            z0 = z
            u0 = u
            x0 = np.clip(x0, 0, np.inf)
            i += 1

        print("ADMM finished after {} iterations".format(i))

        return x0.reshape(shape)

    def LASSO(self, tau=None, beta=1):
        self.prox_f = PO().l11
        self.prox_g = PO(prox_g=PO().l2, y=self.b).translation
        norm = self.powerIteration()
        if tau is not None:
            self.tau = tau
        self.mu = 0.95 * self.tau / norm
        return self.ADMM()

    def powerIteration(self):
        b0 = np.random.rand(len(self.x0.flatten()))
        for _ in range(100):
            mult = self.A.applyAdjoint(self.A.apply(b0)).flatten()
            b = mult / np.linalg.norm(mult)
            sigma = b.T @ b0 / (b0.T @ b0)
            b0 = b
        return sigma
