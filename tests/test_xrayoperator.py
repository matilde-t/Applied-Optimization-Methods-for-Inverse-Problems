import aomip
import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def test_first():
    vol_shape = [128, 128]
    sino_shape = [128]
    thetas = np.arange(360)

    d2c = vol_shape[0] * 100.0
    c2d = vol_shape[0] * 5.0

    phantom = elsa.phantoms.modifiedSheppLogan(vol_shape)
    sinogram = aomip.radon(phantom, sino_shape, thetas, d2c, c2d)
    plt.imshow(sinogram, cmap="gray")
    plt.show()
