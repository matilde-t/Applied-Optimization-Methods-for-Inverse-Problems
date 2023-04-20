from aomip import XrayOperator

import numpy as np
import scipy.io as spio


def loadmat(filename):
    """
    Use SciPy load to load the matlab file and turn it to a Python dictionary.
    Credit to: https://stackoverflow.com/a/8832212
    """

    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            else:
                d[strg] = elem
        return d

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def load_htc2022data(filename, arc=360, arcstart=0, dtype="float32"):
    """Load matlab file and setup XrayProjector and sinogram"""
    # read in matlab file
    mat = loadmat(filename)

    dataset_name = "CtDataFull"
    params = mat[dataset_name]["parameters"]

    # read important parameters
    ds2c = params["distanceSourceOrigin"]
    ds2d = params["distanceSourceDetector"]
    dc2d = ds2d - ds2c

    detpixel_spacing = params["pixelSizePost"]
    num_detpixel = params["numDetectorsPost"]
    angles = params["angles"]

    vol_shape = [512] * 2
    sino_shape = [num_detpixel]

    A = XrayOperator(
        vol_shape,
        sino_shape,
        angles[arcstart : arcstart + (arc * 2)],
        ds2c,
        dc2d,
        vol_spacing=[detpixel_spacing] * 2,
        sino_spacing=[detpixel_spacing],
        dtype=dtype,
    )

    sinogram = mat[dataset_name]["sinogram"].astype(dtype)
    sinogram = prep_sinogram(sinogram)

    return sinogram, A


def prep_sinogram(sino):
    """Necessary procressing of raw data to work with elsa"""
    sino = sino.transpose(1, 0)
    sino = np.flip(sino, axis=1)
    return sino
