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
    return np.mean(x[0:127][0:127])


def capTransmission(x, I0):
    """
    Every value above I0 is set to I0.
    """
    x[x > I0] = I0
    return x
