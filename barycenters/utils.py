import numpy as np


def project_simplex(x):
    """Project Simplex

    Projects an arbitrary vector :math:`\mathbf{x}` into the probability simplex, such that,

    .. math:: \tilde{\mathbf{x}}_{i} = \dfrac{\mathbf{x}_{i}}{\sum_{j=1}^{n}\mathbf{x}_{j}}

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Numpy array of shape (n,)

    Returns
    -------
    y : :class:`numpy.ndarray`
        numpy array lying on the probability simplex of shape (n,)
    """
    x[x < 0] = 0
    if np.isclose(sum(x), 0):
        y = np.zeros_like(x)
    else:
        y = x.copy() / sum(x)
    return y