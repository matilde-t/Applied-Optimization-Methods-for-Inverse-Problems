import numpy as np

def vPad(image, offset):
    """
    Pad the image vertically.
    """
    return np.pad(image, [(offset, offset), (0, 0)], mode="constant")

def hPad(image, offset):
    """
    Pad the image horizontally.
    """
    return np.pad(image, [(0, 0), (offset, offset)], mode="constant")
    