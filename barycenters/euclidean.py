import numpy as np


def fixed_support_barycenter(B):
    """Fixed Support Barycenter

    Calculates the barycenter of a set of N measures in fixed grid X by minimizing the Euclidean distance between
    probability measures. This has closed form,

    .. math:: \mu = \sum_{i=1}^{N}\lambda_{i}a_{i}
    
    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """

    return np.mean(B, axis=0)