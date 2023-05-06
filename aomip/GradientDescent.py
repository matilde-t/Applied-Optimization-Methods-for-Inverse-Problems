import numpy as np

def gradDesc(df, x0, l=1, nmax=1000, eps=1e-6):
    """
    Gradient descent.
    """
    i = 0
    err = np.inf
    while i < nmax and err > eps:
        x = x0 - l * df(x0)
        err = np.linalg.norm(x - x0)
        x0 = x
        i = i+1
    print('Number of iterations: {}'.format(i))
    return x