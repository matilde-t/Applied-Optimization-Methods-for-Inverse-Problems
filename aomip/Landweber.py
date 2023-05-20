import numpy as np


class LW:
    def __init__(self, A, b, x0, nmax=1000, eps=1e-6, l=None):
        self.A = A
        self.b = b
        self.x0 = x0
        self.nmax = nmax
        self.eps = eps
        self.l = l

    def solve(self, l=None):
        """
        Implement the Ladweber iteration.
        """
        print("Starting Landweber iteration")
        self.l = l
        if self.l is None:
            self.l = 1 / self.sigma2()
        x0 = self.x0.copy()
        x0 = x0
        for i in range(self.nmax):
            x = x0 - self.l * self.A.applyAdjoint(self.A.apply(x0) - self.b)
            if np.linalg.norm(x - self.x0) < self.eps:
                break
            x0 = x
        print("Number of iterations: {}".format(i + 1))
        return x.reshape(self.x0.shape)

    def sigma2(self, b0=None, nmax=100):
        """
        Compute the largest singular value of A through power itearations.
        """
        if b0 is None:
            b0 = np.zeros(len(self.x0.flatten()))
            b0[0] = 1
        for i in range(nmax):
            Cb = self.A.applyAdjoint(self.A.apply(b0))
            b = Cb / np.linalg.norm(Cb)
            b0 = b
        proj = np.dot(b.flatten(), self.A.applyAdjoint(self.A.apply(b)).flatten())
        return proj / np.linalg.norm(b)
