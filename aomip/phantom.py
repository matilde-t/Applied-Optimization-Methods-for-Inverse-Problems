import pyelsa as elsa
import numpy as np

def shepp_logan(size):
    return np.rot90(elsa.phantoms.modifiedSheppLogan(size), -1)
