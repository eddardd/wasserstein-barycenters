import ot
import numpy as np

from barycenters import project_simplex


def fixed_support_barycenter(B, M, weights=None, eta=10, numItermax=100, stopThr=1e-9, verbose=False, norm='max'):
    """Fixed Support Wasserstein Barycenter

    We follow the Algorithm 1. of [1], into calculating the Wasserstein barycenter of N measures over a pre-defined
    grid :math:`\mathbf{X}`. These measures, of course, have variable sample weights :math:`\mathbf{b}_{i}`.
    
    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.
    M : :class:`numpy.ndarray`
        Numpy array of shape (d, d), containing the pairwise distances for the support of B
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
    norm : str
        Either 'max', 'median' or 'none'. If not 'none', normalizes pairwise distances.

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """
    a = ot.unif(B.shape[1])
    a_prev = a.copy()
    weights = ot.unif(B.shape[0]) if weights is None else weights
    if norm == "max":
        _M = M / np.max(M)
    elif norm == "median":
        _M = M / np.median(M)
    else:
        _M = M

    for k in range(numItermax):
        potentials = []
        for i in range(B.shape[0]):
            _, ret = ot.emd(a, B[i], _M, log=True)
            potentials.append(ret['u'])
        
        # Calculates the gradient
        f_star = sum(potentials) / len(potentials)

        # Mirror Descent
        a = a * np.exp(- eta * f_star)

        # Projection
        a = project_simplex(a)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))

        # Update previous a
        a_prev = a.copy()
    return a