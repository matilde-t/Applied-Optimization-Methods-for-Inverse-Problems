import numpy as np
from .ProximalOperators import ProximalOperators as PO
from .ForwardDifferences import FirstDerivative


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
        sigma=1,
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.mu = mu
        self.tau = tau
        self.nmax = nmax
        self.sigma = sigma

    def ADMM(self):
        """
        ADMM algorithm.
        """
        print("Starting ADMM")

        i = 0
        shape = self.x0.shape
        x0 = self.x0.copy()
        z0 = self.A.apply(x0)
        u0 = np.zeros_like(z0)

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

    def LASSO(self, tau=None, beta=1, sigma=1):
        if sigma is not None:
            self.sigma = sigma
        self.prox_f = PO(l=beta, sigma=self.sigma).l11
        self.prox_g = PO(
            prox_g=PO(sigma=self.sigma).l2, y=self.b, sigma=self.sigma
        ).translation
        norm = self.powerIteration()
        if tau is not None:
            self.tau = tau
        self.mu = 0.95 * self.tau / norm
        return self.ADMM()

    def TV_anisotropic(self, tau=None, beta=1, sigma=1):
        if sigma is not None:
            self.sigma = sigma
        if tau is not None:
            self.tau = tau
        norm = 1e7  # self.powerIteration()
        self.mu = 0.95 * self.tau / norm
        K = StackedOperator([self.A, FirstDerivative()])
        self.A = K
        h_prox = [
            PO(prox_g=PO(sigma=sigma).l21, y=self.b, sigma=sigma).translation,
            PO(l=beta, sigma=sigma).l11,
        ]
        self.prox_f = PO(sigma=sigma).constant
        self.prox_g = prox_sep(h_prox, sigma=sigma).prox
        return self.ADMM()

    def TV_isotropic(self, tau=None, beta=1, sigma=1):
        if sigma is not None:
            self.sigma = sigma
        if tau is not None:
            self.tau = tau
        norm = 1e7  # self.powerIteration()
        self.mu = 0.95 * self.tau / norm
        K = StackedOperator([self.A, FirstDerivative()])
        self.A = K
        h_prox = [
            PO(prox_g=PO(sigma=sigma).l21, y=self.b, sigma=sigma).translation,
            PO(l=beta, sigma=sigma).l21,
        ]
        self.prox_f = PO(sigma=sigma).constant
        self.prox_g = prox_sep(h_prox, sigma=sigma).prox
        return self.ADMM()

    def powerIteration(self):
        b0 = np.random.rand(len(self.x0.flatten()))
        for _ in range(100):
            mult = self.A.applyAdjoint(self.A.apply(b0)).flatten()
            b = mult / np.linalg.norm(mult)
            sigma = b.T @ b0 / (b0.T @ b0)
            b0 = b
        return sigma


class StackedOperator:
    def __init__(self, operators):
        self.ops = operators

    def apply(self, x):
        l = np.array([], dtype=object)
        l.resize(len(self.ops), refcheck=False)
        for i, op in enumerate(self.ops):
            l[i] = op.apply(x)
        return l

    def applyAdjoint(self, y):  # now y is a list
        x = self.ops[0].applyAdjoint(y[0])
        for yi, op in zip(y[1:], self.ops[1:]):
            x += op.applyAdjoint(yi)
        return x


class separable_sum:
    def __init__(self, h):
        self.h = h

    def sum(self, x):
        tot = np.zeros_like(x)
        for f in self.h:
            tot += f(x)
        return tot


class prox_sep:
    def __init__(self, h_prox, sigma=1):
        self.h_prox = h_prox
        self.sigma = sigma

    def prox(self, x):
        res = np.zeros_like(x)
        for i in range(len(self.h_prox)):
            res[i] = self.h_prox[i](x[i], self.sigma)
        return res
