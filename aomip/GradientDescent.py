import numpy as np
import scipy as sp


def gradDesc(df, x0, l=1e-3, nmax=1000, eps=1e-6):
    """
    Gradient descent.
    """
    print("Starting gradient descent")
    i = 0
    err = np.inf
    shape = x0.shape
    x0 = x0.flatten()
    while i < nmax and err > eps:
        x = x0 - l * df(x0)
        err = np.linalg.norm(x - x0)
        x0 = x
        i = i + 1
    print("Number of iterations: {}".format(i))
    return x.reshape(shape)


def leastSquares(A, b, x0, l=1e-3, nmax=1000, eps=1e-6):
    """
    Solve the Least Squares Problem using Gradient Descent.
    """
    df = lambda x: A.applyAdjoint(A.apply(x) - b).flatten()
    return gradDesc(df, x0, l, nmax, eps)


def l2Norm(A, b, x0, L=None, beta=1, l=1e-3, nmax=1000, eps=1e-6):
    """
    Solves the Tikhonov problem in the form (1/2)||Ax-b||^2 + (beta/2)l^2||Lx||^2
    using gradient descent.
    """
    if L is None:
        L = sp.sparse.eye(len(x0.flatten()))
    df = (
        lambda x: A.applyAdjoint(A.apply(x) - b).flatten()
        + beta / 2 * L.T @ L @ x.flatten()
    )
    return gradDesc(df, x0, l, nmax, eps)


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


def huber(A, b, x0, delta=1, beta=1, l=1e-3, nmax=1000, eps=1e-6):
    """
    Solves the Tikhonov problem with l1 regularization.
    """
    df = lambda x: A.applyAdjoint(A.apply(x) - b) + beta / 2 * Ld(x, delta)
    return gradDesc(df, x0, l, nmax, eps)


def fair(A, b, x0, delta=1, beta=1, l=1e-3, nmax=1000, eps=1e-6):
    """
    Solves the Tikhonov problem with Fair potential.
    """
    df = lambda x: A.applyAdjoint(A.apply(x) - b) + beta / 2 * x / (1 + x / delta)
    return gradDesc(df, x0, l, nmax, eps)


def forwardDiff(x):
    """
    Construct forward difference operator in 2D
    """

    len = np.prod(x.shape)
    n = x.shape[0]
    dx = sp.sparse.diags([-1, 1], [0, 1], (len, len))
    dy = sp.sparse.diags([-1, 1], [0, n + 1], (len, len))
    return sp.sparse.vstack([dx, dy])
