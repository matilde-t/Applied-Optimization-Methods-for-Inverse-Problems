import aomip
import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt

def test_shepp_logan():
    """Do a forward and backward projection of a complex phantom"""

    vol_shape = [128, 128]
    sino_shape = [128]
    thetas = np.arange(360)

    d2c = vol_shape[0] * 100.0
    c2d = vol_shape[0] * 5.0

    x = aomip.phantom.shepp_logan(vol_shape)
    A = aomip.XrayOperator(vol_shape, sino_shape, thetas, d2c, c2d)

    sinogram = A.apply(x)
    backprojection = A.applyAdjoint(sinogram)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(x, cmap="gray")
    ax1.set_title("Original image/volume/phantom")
    ax2.imshow(sinogram, cmap="gray")
    ax2.set_title("Sinogram or Forward projection")
    ax3.imshow(backprojection, cmap="gray")
    ax3.set_title("Back projection")

    plt.show()

def test_square_phantom_apply():
    """Do a forward and backprojection of a small phantom using the interface provided by the XrayOperator"""

    vol_shape = [128, 128]
    sino_shape = [128]
    thetas = np.arange(360)

    d2c = vol_shape[0] * 100.0
    c2d = vol_shape[0] * 5.0

    x = np.zeros(vol_shape)
    x[40:60, 40:60] = 1

    A = aomip.XrayOperator(vol_shape, sino_shape, thetas, d2c, c2d)

    sinogram = A.apply(x)
    backprojection = A.applyAdjoint(sinogram)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(x, cmap="gray")
    ax1.set_title("Original image/volume/phantom")
    ax2.imshow(sinogram, cmap="gray")
    ax2.set_title("Sinogram or Forward projection")
    ax3.imshow(backprojection, cmap="gray")
    ax3.set_title("Back projection")

    plt.show()

def test_square_phantom():
    """Do a forward and backprojection of a small phantom using the interface provided by scipy.LinearOpertor"""

    vol_shape = [128, 128]
    sino_shape = [128]
    thetas = np.arange(360)

    d2c = vol_shape[0] * 100.0
    c2d = vol_shape[0] * 5.0

    x = np.zeros(vol_shape)
    x[40:60, 40:60] = 1

    A = aomip.XrayOperator(vol_shape, sino_shape, thetas, d2c, c2d)

    sinogram = (A * x.flatten()).reshape((sino_shape[-1], len(thetas)))
    backprojection = (A.H * sinogram.flatten()).reshape(vol_shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(x, cmap="gray")
    ax1.set_title("Original image/volume/phantom")
    ax2.imshow(sinogram, cmap="gray")
    ax2.set_title("Sinogram or Forward projection")
    ax3.imshow(backprojection, cmap="gray")
    ax3.set_title("Back projection")

    plt.show()


if __name__ == "__main__":
    test_shepp_logan()
    test_square_phantom()
