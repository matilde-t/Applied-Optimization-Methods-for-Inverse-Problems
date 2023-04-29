import numpy as np


def centerOfRotationCorrection(image, pixels, axis):
    return np.roll(image, pixels, axis)
