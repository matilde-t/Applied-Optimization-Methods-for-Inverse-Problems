import numpy as np

def slice(img, idx):
    """Slice a 3D projection 
    and extract a 2D sinogram."""

    sino = []
    for projection in img:
        sino.append(projection[idx,:])
    return np.array(sino).T