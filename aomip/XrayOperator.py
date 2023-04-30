"""
"""

import pyelsa as elsa
from scipy.sparse.linalg import LinearOperator
import numpy as np


class XrayOperator(LinearOperator):
    r"""X-ray Operator

    Apply the 2- or 3-dimensional X-ray transform (and adjoint) to the signal

    By applying the operator directly to a vector (often referred to as forward
    projection), it is the simulation of the phsyical process of X-rays
    traversing an object. The result is a so called sinogram. Is is the stack of
    X-ray shadows of the desired object from all given projection angles (this
    version only supports circular trajectories).

    The adjoint operation is the so called backward projection. It takes the
    sinogram as an input, and creates an object in the image/volume domain.

    Given the `m x n` sized X-ray operator, `m` is equal to the number of
    detector pixels times the number of projection angles. `n` is the number of
    pixels/voxels in the image/volume.

    Parameters
    ----------
    vol_shape : :obj:`np.ndarray`
        Size of the image/volume
    sino_shape : :obj:`np.ndarray`
        Size of the sinogram
    thetas : :obj:`np.ndarray`
        List of projection angles in degree
    s2c : :obj:`float32`
        Distance from source to center of rotation
    c2d : :obj:`float32`
        Distance from center of rotation to detector
    vol_spacing : :obj:`np.ndarray`, optional
        Spacing of the image/volume, i.e. size of each pixel/voxel. By default
        unit size is assumed.
    sino_spacing : :obj:`np.ndarray`, optional
        Spacing of the sinogram, i.e. size of each detector pixel. By default
        unit size is assumed.
    cor_offset : :obj:`np.ndarray`, optional
        Offset of the center of rotation. By default no offset is applied.
    pp_offset : :obj:`np.ndarray`, optional
        Offset of the principal point. By default no offset is applied.
    projection_method : :obj:`str`, optional
        Projection method used for the forward and backward projections. By
        default the interpolation/Joseph's method ('josephs') is used. Can also
        be 'siddons', for the line intersection length methods often referred
        to as Siddons method.
    dtype : :obj:`float32`, optional
        Type of elements in input array.
    """

    def __init__(
        self,
        vol_shape,
        sino_shape,
        thetas,
        s2c,
        c2d,
        vol_spacing=None,
        sino_spacing=None,
        cor_offset=None,
        pp_offset=None,
        projection_method="josephs",
        dtype="float32",
    ):
        self.vol_shape = np.array(vol_shape)
        self.sino_shape = np.array(sino_shape)

        # Sinogram is of the same dimension as volume (i.e. it's a stack
        # of (n-1)-dimensional projection)
        if self.vol_shape.size != (self.sino_shape.size + 1):
            raise RuntimeError(
                f"Volume and sinogram must be n and (n-1) dimensional (is {self.vol_shape.size} and {self.sino_shape.size})"
            )

        self.ndim = np.size(vol_shape)

        self.thetas = np.array(thetas)

        # thetas needs to be a 1D array / list
        if self.thetas.ndim != 1:
            raise RuntimeError(
                f"angles must be a 1D array or list (is {self.thetas.ndim})"
            )

        self.s2c = s2c
        self.c2d = c2d

        self.vol_spacing = (
            np.ones(self.ndim) if vol_spacing is None else np.array(vol_spacing)
        )
        self.sino_spacing = (
            np.ones(self.ndim - 1) if sino_spacing is None else np.array(sino_spacing)
        )
        self.cor_offset = (
            np.zeros(self.ndim) if cor_offset is None else np.array(cor_offset)
        )
        self.pp_offset = (
            np.zeros(self.ndim - 1) if pp_offset is None else np.array(pp_offset)
        )

        # Some more sanity checking
        if self.vol_spacing.size != self.ndim:
            raise RuntimeError(
                f"Array containing spacing of volume is of the wrong size (is {self.vol_spacing.size}, expected {self.ndim})"
            )

        if self.cor_offset.size != self.ndim:
            raise RuntimeError(
                f"Array containing offset of center of rotation is of the wrong size (is {self.cor_offset.size}, expected {self.ndim})"
            )

        if self.sino_spacing.size != self.ndim - 1:
            raise RuntimeError(
                f"Array containing spacing of detector is of the wrong size (is {self.sino_spacing.size}, expected {self.ndim - 1})"
            )

        if self.pp_offset.size != self.ndim - 1:
            raise RuntimeError(
                f"Array containing principal point offset is of the wrong size (is {self.pp_offset.size}, expected {self.ndim - 1})"
            )

        self.vol_descriptor = elsa.VolumeDescriptor(self.vol_shape, self.vol_spacing)
        self.sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
            thetas,
            self.vol_descriptor,
            self.s2c,
            self.c2d,
            self.pp_offset,
            self.cor_offset,
            self.sino_shape,
            self.sino_spacing,
        )

        if projection_method == "josephs":
            if elsa.cudaProjectorsEnabled():
                self.A = elsa.JosephsMethodCUDA(
                    self.vol_descriptor, self.sino_descriptor
                )
            else:
                self.A = elsa.JosephsMethod(self.vol_descriptor, self.sino_descriptor)
        elif projection_method == "siddons":
            if elsa.cudaProjectorsEnabled():
                self.A = elsa.SiddonsMethodCUDA(
                    self.vol_descriptor, self.sino_descriptor
                )
            else:
                self.A = elsa.SiddonsMethod(self.vol_descriptor, self.sino_descriptor)
        else:
            raise RuntimeError(f"Unknown projection method '{projection_method}'")

        M = np.prod(sino_shape) * np.size(thetas)  # number of rows
        N = np.prod(vol_shape)  # number of columns

        self.dtype = dtype
        self.shape = (M, N)

        super().__init__(self.dtype, self.shape)

    def apply(self, x):
        """Apply the forward projection to x

        Perform or simulate the forward projection given the specified parameters.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Object to forward project
        """
        # copy/move numpy array to elsa
        ex = elsa.DataContainer(
            np.reshape(x, self.vol_shape, order="C"), self.vol_descriptor
        )

        # perform forward projection
        sino = self.A.apply(ex)

        # return a numpy array
        return np.array(sino)

    def applyAdjoint(self, sino):
        """Apply the back projection to sino

        Perform or simulate the back projection given the specified parameters.

        The returned array is a 1D-vector containing the backprojection, which
        can be recreated using `backprojection.reshape(shape, order="F")`. Where
        shape the volume/image size.

        Parameters
        ----------
        sino : :obj:`np.ndarray`
            Sinogram to back project
        """
        # copy/move sinogram to elsa
        shape = np.concatenate((self.sino_shape, np.array([np.size(self.thetas)])))
        esino = elsa.DataContainer(
            np.reshape(sino, shape, order="C"), self.sino_descriptor
        )

        # perform backward projection
        bp = self.A.applyAdjoint(esino)

        # return a numpy array
        return np.array(bp) / len(self.thetas)

    def _matvec(self, x):
        """Perform the forward projection, implement the scipy.LinearOperator interface"""
        return self.apply(x).flatten("C")

    def _adjoint(self):
        """Return the adjoint, implement the scipy.LinearOperator interface"""

        class AdjointXrayOperator(LinearOperator):
            def __init__(self, forward):
                self.forward = forward
                super().__init__(self.forward.dtype, np.flip(self.forward.shape))

            def _matvec(self, sino):
                return self.forward.applyAdjoint(sino).flatten("C")

            def _adjoint(self):
                return self.forward

        return AdjointXrayOperator(self)


def radon(
    x,
    sino_shape,
    thetas,
    s2c,
    c2d,
    vol_spacing=None,
    sino_spacing=None,
    cor_offset=None,
    pp_offset=None,
    projection_method="josephs",
    dtype="float32",
):
    """Forward project x into Radon space

    For details on the arguments see the `XrayOperator`
    """
    x = np.array(x)

    A = XrayOperator(
        x.shape,
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

    sinogram_flat = A * x.flatten()

    shape = np.concatenate((sino_shape, np.array([np.size(thetas)])))
    return sinogram_flat.reshape(shape)
