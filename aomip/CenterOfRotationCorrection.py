import numpy as np


def centerOfRotationCorrection(image, pixels, axis):
    """
    This function takes an image and shifts it by a given number of pixels
    along a given axis.
    positive nmber of pixels shifts the image to the right
    negative number of pixels shifts the image to the left
    axis = 0 shifts the image along the x-axis
    axis = 1 shifts the image along the y-axis"""

    return np.roll(image, pixels, axis)
