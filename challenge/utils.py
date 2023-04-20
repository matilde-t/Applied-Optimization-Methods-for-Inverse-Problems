from aomip import XrayOperator

import numpy as np
import scipy.io as spio
from skimage.filters import threshold_otsu


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
    sinogram = prep_sinogram(sinogram)[:, : (arc * 2)]

    return sinogram, A


def prep_sinogram(sino):
    """Necessary procressing of raw data to work with elsa"""
    sino = sino.transpose(1, 0)
    sino = np.flip(sino, axis=1)
    return sino


def segment(img):
    """Create segmented binary image using otsu thresholding"""
    img_seg = img.copy()
    img_seg[img_seg < 0] = 0
    thresh = threshold_otsu(img_seg)
    img_seg[img < thresh] = 0
    img_seg[img > thresh] = 1

    img_seg = img_seg.astype(bool)

    return img_seg


def calculate_score(recon, groundthruth):
    """Compute score of the reconstruction comapred to the ground truth.

    The score will be in the range of [-1, 1], where 1 (best) represents a
    perfect reconstruction, 0 is equivalent to a random reconstruction,
    and -1 indicates absolute disagreement.

    The score is computed using morphological closing, a confusion matrix and
    Matthews correlation coefficient. This is the code provided by the Helsinki
    Tomography challenge to compute the score.

    Parameters
    ----------
    recon : :obj:`np.ndarray`
        Binary segmentation of the reconstruction to calculate the score to.
    groundthruth : :obj:`np.ndarray`
        Binary segmentation of the ground truth reconstruction
    """

    def AND(x, y):
        return np.logical_and(x, y)

    def NOT(x):
        return np.logical_not(x)

    # if the reconstruction does not have the correct size, score is 0
    if recon.shape != (512, 512):
        return 0

    # confusion matrix with true positives, true negatives, false positives, and
    # false negatives
    TP = float(len(np.where(AND(groundthruth, recon))[0]))
    TN = float(len(np.where(AND(NOT(groundthruth), NOT(recon)))[0]))
    FP = float(len(np.where(AND(NOT(groundthruth), recon))[0]))
    FN = float(len(np.where(AND(groundthruth, NOT(recon)))[0]))
    cmat = np.array([[TP, FN], [FP, TN]])

    # Matthews correlation coefficient (MCC)
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        score = 0
    else:
        score = numerator / denominator

    return score
