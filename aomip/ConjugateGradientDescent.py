import numpy as np


def CGD(A_, b_, x0, nmax=3, eps=1e-5):
    """
    Conjugate Gradient Descent.
    """
    print("Starting conjugate gradient descent")
    A = lambda x: A_.applyAdjoint(A_.apply(x))
    b = A_.applyAdjoint(b_)
    r0 = b - A(x0)
    p = r0
    x = x0
    k = 0
    while np.linalg.norm(r0) > eps and k < nmax:
        alpha = (r0.T @ r0) / (p.T @ A(p))
        x = x + alpha * p
        r = r0 - alpha * A(p)
        beta = (r.T @ r) / (r0.T @ r0)
        p = r + beta * p
        r0 = r
        k = k + 1
    print("Number of iterations: {}".format(k))
    return x
