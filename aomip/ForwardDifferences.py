import numpy as np

def forward_diff(x, axis=-1, sampling=1):
    """Apply forward differences to any chosen axis of the given input array
    Notes
    -----
    For simplicity, given a one dimensional array, the first-order forward
    stencil is:
    .. math::
    y[i] = (x[i+1] - x[i]) / \Delta x
    """
    # swap the axis we want to calculate finite differences on,
    # to the last dimension
    x = x.swapaxes(axis, -1)
    # create output vector
    y = np.zeros_like(x)
    # compute finite differences
    y[..., :-1] = (x[..., 1:] - x[..., :-1]) / sampling
    # swap axis back to original position
    return y.swapaxes(axis, -1)

def adjoint_forward_diff(x, axis=-1):
    """Apply the adjoint of the forward differences to any chosen axis of
    the given input array
    """
    x = x.swapaxes(axis, -1)
    y = np.zeros_like(x)
    y[..., :-1] -= x[..., :-1]
    y[..., 1:] += x[..., :-1]
    return y.swapaxes(axis, -1)

class FirstDerivative:
    """Operator which applies finite differences to the given dimensions
    Notes
    -----
    This operator computed the finite differences lazily, i.e. doesn't construct
    a matrix and consumes unnecessary memory
    """
    def apply(self, x):
        dim = x.ndim
        grad = np.zeros((dim,) + x.shape)
        for i in range(dim):
            grad[i, ...] = forward_diff(x, axis=i).copy()
        return grad

    def applyAdjoint(self, x):
        grad = np.zeros(x.shape[1:])
        for i in range(x.shape[0]):
            grad[...] += adjoint_forward_diff(x[i], axis=i).copy()
        return grad
