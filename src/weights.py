import numpy as np
from normalisations import minmax_normalization
from helpers import correlation_matrix, pearson


def critic_weights(matrix, *args, **kwargs):
    """ Calculate weights for given `matrix` using CRITIC method.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

        Returns
        -------
            ndarray
                Vector of weights.
    """
    # nmatrix = normalize_matrix(matrix, minmax_normalization, None)
    nmatrix = minmax_normalization(matrix)
    std = np.std(nmatrix, axis=0, ddof=1)
    coef = correlation_matrix(nmatrix, pearson, True)
    C = std * np.sum(1 - coef, axis=0)
    return C / np.sum(C)
