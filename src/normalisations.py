import numpy as np


def minmax_normalization(x, cost=False):
    """Calculate the normalized vector using the min-max method.

    Parameters
    ----------
        x : ndarray
            One-dimensional numpy array of values to be normalized

        cost : bool, optional
            Vector type. Default profit type.

    Returns
    -------
        ndarray
            One-dimensional numpy array of normalized values.
    """
    if np.min(x) == np.max(x):  # If all values are equal
        return np.ones(x.shape)

    if cost:
        return (np.max(x) - x) / (np.max(x) - np.min(x))
    return (x - np.min(x)) / (np.max(x) - np.min(x))
