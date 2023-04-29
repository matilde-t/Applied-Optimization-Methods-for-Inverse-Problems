import numpy as np


def transmissionToAbsorption(x, I0):
    """
    Convert transmission image into absorption image.
    """
    x = capI0(x, I0)
    return -np.log(x / I0)


def absorptionToTransmission(x, I0):
    """
    Convert absorption image into transmission image.
    """
    x = cap0(x)
    return I0 - transmissionToAbsorption(x, I0)


def findI0(x):
    """
    Find the value of I0 for a given image.
    """
    return np.mean(np.max(x, axis=0))


def capI0(x, I0):
    """
    Cap the value of I0 to be at least 1.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > I0:
                x[i][j] = I0
    return x


def cap0(x):
    """
    Cap the value of x to be at least 0.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                x[i][j] = 0
    return x
