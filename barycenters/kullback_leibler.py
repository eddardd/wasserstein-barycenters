import ot
import numpy as np
from barycenters import project_simplex


def fixed_support_barycenter(B, eta=1, numItermax=100, stopThr=1e-9, verbose=True):
    """Fixed Support Barycenter

    Calculates the barycenter over a set of N measures using the Kullback-Leibler divergence

    .. math:: KL(p||q) = \sum_{i=1}^{n}p_{i}\log\dfrac{p_{i}}{q_{i}}

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.
    weights : :class:`numpy.ndarray`
        Numpy array or None. If None, weights are uniform. Otherwise, weights each measure in the barycenter
    eta : float
        Mirror descent step size
    numItermax : integer
        Maximum number of descent steps
    stopThr : float
        Threshold for stopping mirror descent iterations
    verbose : bool
        If true, display information about each descent step

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """

    a = ot.unif(B.shape[1])
    a_prev = a.copy()

    for k in range(numItermax):
        G = []
        for i in range(B.shape[0]):
            _b = B[i].copy()
            _b[_b == 0] = 1
            div = a / _b
            div[div < 0] = 0
            G.append(np.log(div))
        g = sum(G) / len(G)

        a = project_simplex(a - eta * g)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))
        a_prev = a.copy()

    return a