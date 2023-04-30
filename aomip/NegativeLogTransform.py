import numpy as np


def getTransmission(I1, I0):
    """
    Convert image into transmission image.
    """
    return capTransmission(I1, I0) / I0


def getAbsorption(I1, I0):
    """
    Convert image into absorption image.
    """
    return -np.log(capTransmission(I1, I0) / I0)


def transmissionToAbsorption(T):
    """
    Convert transmission image into absorption image.
    """
    return -np.log(T)


def absorptionToTransmission(A):
    """
    Convert absorption image into transmission image.
    """
    return np.exp(-A)


def findI0(x):
    """
    Find the value of I0 for a given image.
    """
    return np.mean(np.max(x, axis=0))


def capTransmission(x, I0):
    """
    Every value above I0 is set to I0.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > I0:
                x[i][j] = I0
    return x
