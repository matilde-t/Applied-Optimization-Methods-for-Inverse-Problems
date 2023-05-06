from aomip import gradDesc

def leastSquares(A, b, x0, l=1, nmax=1000, eps=1e-6):
    """
    Solve the Least Squares Problem using Gradient Descent.
    """
    df = lambda x: A.applyAdjoint(A.apply(x) - b)
    return gradDesc(df, x0, l, nmax, eps)