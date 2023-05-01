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
    sino = np.fft.fftn(sino)
    x_freq = np.fft.fftfreq(sino.shape[0])
    y_freq = np.fft.fftfreq(sino.shape[1])

    if filter == "ram-lak":
        for i in range(x_freq.shape[0]):
            for j in range(y_freq.shape[0]):
                if x_freq[i] < -0.5 or x_freq[i] > 0.5:
                    sino[i][:] = 0
                elif y_freq[j] < -0.5 or y_freq[j] > 0.5:
                    sino[:][j] = 0
                else:
                    sino[i][j] = np.sqrt(x_freq[i] ** 2 + y_freq[j] ** 2) * sino[i][j]
    elif filter == "shepp-logan":
        for i in range(x_freq.shape[0]):
            for j in range(y_freq.shape[0]):
                if x_freq[i] < -0.5 or x_freq[i] > 0.5:
                    sino[i][:] = 0
                elif y_freq[j] < -0.5 or y_freq[j] > 0.5:
                    sino[:][j] = 0
                else:
                    sino[i][j] = (
                        4
                        * np.sqrt(x_freq[i] ** 2 + y_freq[j] ** 2)
                        * np.sin(np.pi * x_freq[i] / 2)
                        * np.sin(np.pi * y_freq[j] / 2)
                        * sino[i][j]
                    ) / (x_freq[i] * y_freq[j] * np.pi**2)
    elif filter == "cosine":
        for i in range(x_freq.shape[0]):
            for j in range(y_freq.shape[0]):
                if x_freq[i] < -0.5 or x_freq[i] > 0.5:
                    sino[i][:] = 0
                elif y_freq[j] < -0.5 or y_freq[j] > 0.5:
                    sino[:][j] = 0
                else:
                    sino[i][j] = (
                        np.sqrt(x_freq[i] ** 2 + y_freq[j] ** 2)
                        * np.cos(np.pi * x_freq[i] / 2)
                        * np.cos(np.pi * y_freq[j] / 2)
                        * sino[i][j]
                    )
    else:
        pass

    return A.applyAdjoint(np.fft.ifftn(sino)).reshape(x_shape, order="F")
