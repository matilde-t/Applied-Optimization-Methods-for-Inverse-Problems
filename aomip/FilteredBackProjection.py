import numpy as np
from aomip import XrayOperator


def iradon(
    sino,
    sino_shape,
    x_shape,
    thetas,
    s2c,
    c2d,
    filter="ram-lak",
    vol_spacing=None,
    sino_spacing=None,
    cor_offset=None,
    pp_offset=None,
    projection_method="josephs",
    dtype="float32",
):
    """Backward project the phantom from Radon space

    For details on the arguments see the `XrayOperator`
    """

    A = XrayOperator(
        x_shape,
        sino_shape,
        thetas,
        s2c,
        c2d,
        vol_spacing=vol_spacing,
        sino_spacing=sino_spacing,
        cor_offset=cor_offset,
        pp_offset=pp_offset,
        projection_method=projection_method,
        dtype=dtype,
    )
    H = np.linspace(-1, 1, sino.shape[0])
    num_angles = len(thetas)
    if filter == "ram-lak":
        H = np.abs(H)
    elif filter == "shepp-logan":
        H = np.abs(H) * np.sinc(H / 2)
    elif filter == "cosine":
        H = np.abs(H) * np.cos(H * np.pi / 2)
    else:
        pass

    h = np.tile(H, (num_angles, 1)).T
    
    fftsino = np.fft.fft(sino, axis=0)
    projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
    fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))

    return A.applyAdjoint(fsino).reshape(x_shape, order="F")
