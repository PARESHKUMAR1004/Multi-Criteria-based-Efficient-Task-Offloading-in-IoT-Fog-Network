import numpy as np


def _cov(x, y):
    return np.cov(x, y, bias=True)[0][1]


def pearson(x, y):
    """Calculate Pearson correlation between two raw vectors.

    Parameters
    ----------
        x : ndarray
            First vector with raw values.

        y : ndarray
            Second vector with raw values.

    Returns
    -------
        float
            Correlation between two vectors.
    """
    return (_cov(x, y)) / (np.std(x) * np.std(y))


def correlation_matrix(rankings, method, columns=False):
    """ Creates a correlation matrix for given vectors from the numpy array.

        Parameters
        ----------
            rankings : ndarray
                Vectors for which the correlation matrix is to be calculated.

            method : callable
                Function to calculate the correlation matrix.

            columns: bool
                If the column value is set to true then the correlation matrix will be calculated for the columns.
                Otherwise the matrix will be calculated for the rows.

        Returns
        -------
            ndarray
                Correlation between two rankings vectors.
    """
    if columns:
        rankings = rankings.T
    n = rankings.shape[0]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = method(rankings[i], rankings[j])
    return corr
