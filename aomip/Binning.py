import numpy as np

def binning(x):
    """
    Binning of image x.
    """
    if len(x.shape) == 1:
        y = np.zeros(x.shape[0]//2)
        for i in range(y.shape[0]):
            y[i] = (x[2*i] + x[2*i+1]) / 2
        return y
    elif len(x.shape) == 2:
        y = np.zeros((x.shape[0]//2, x.shape[1]//2))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i][j] = (x[2*i][2*j] + x[2*i+1][2*j] + x[2*i][2*j+1] + x[2*i+1][2*j+1]) / 4
        return y