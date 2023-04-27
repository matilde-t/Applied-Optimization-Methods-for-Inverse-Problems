import numpy as np

def transmissionToAbsorption(x, I0):
    """
    Convert transmission image into absorption image.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > I0:
                x[i][j] = I0
    return -np.log(x) / I0

def absorptionToTransmission(x, I0):
    """
    Convert absorption image into transmission image.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                x[i][j] = 0
    return -np.log(x) / I0