import json
import tifffile
import numpy as np
from aomip import binning, iradon
from .Slicing import slice
from .XrayOperator import XrayOperator


def load_tiff_stack_with_metadata(file):
    if not (file.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", '"')
    try:
        metadata = json.loads(metadata)
    except:
        print("The tiff file you try to open does not seem to have metadata attached.")
        metadata = None
    return data, metadata


def load_lowdose_data(file, idx):
    data, metadata = load_tiff_stack_with_metadata(file)

    # extract sinogram from data through slicing (binning because the data is too large)
    sino = slice(data, idx)
    sino = binning(sino)
    sino = sino.transpose(1, 0)
    sino = np.flip(sino, axis=1)

    # extract angles in degree
    angles = np.degrees(np.array(metadata["angles"])[: metadata["rotview"]])
    angles = angles[0::2]

    # setup some spacing and sizes
    voxel_size = 0.7  # can be adjusted
    vox_scaling = 1 / voxel_size
    vol_spacing = [vox_scaling] * 2

    # size of detector
    det_count = sino.shape[:-1]
    det_spacing = vox_scaling * metadata["du"]

    # distances from source to center, and center to detector
    ds2c = vox_scaling * metadata["dso"]
    dc2d = vox_scaling * metadata["ddo"]

    vol_shape = [512] * 2
    sino_shape = [det_count[0]]

    # setup XrayOperator
    A = XrayOperator(
        vol_shape,
        sino_shape,
        angles,
        ds2c,
        dc2d,
        vol_spacing=vol_spacing,
        sino_spacing=[det_spacing],
        dtype="float32",
    )

    # initial reconstructed image
    img = iradon(
        sino,
        sino_shape,
        vol_shape,
        angles,
        ds2c,
        dc2d,
        vol_spacing=vol_spacing,
        sino_spacing=[det_spacing],
        dtype="float32",
    )

    return sino, A, img
