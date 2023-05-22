import numpy as np

def landweber(A, b, x0, nmax=1000, eps=1e-6, l=None):
    """
    Implement the Ladweber iteration.
    """
    print("Starting Landweber iteration")
    if l is None:
        l = 1 / sigma2(A)
    for i in range(nmax):
        x = x0 - l * A.T @ (A @ x0 - b)
        if np.linalg.norm(x - x0) < eps:
            break
        x0 = x
    print("Number of iterations: {}".format(i + 1))
    return x

def sigma2(A, b0=None, nmax=100):
    """
    Compute the largest singular value of A through power itearations.
    """
    if b0 is None:
        b0 = np.zeros(A.shape[1])
        b0[0] = 1
    for i in range(nmax):
        Cb = A.T @ (A @ b0)
        b = Cb / np.linalg.norm(Cb)
        b0 = b
    proj = np.dot(b, A.T @ (A @ b))
    return proj / np.linalg.norm(b)