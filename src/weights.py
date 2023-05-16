import numpy as np
from normalisations import minmax_normalization, sum_normalization
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
    coef = correlation_matrix(nmatrix, pearson, True) #coef= corelation matrix of the normalised matrix
    C = std * np.sum(1 - coef, axis=0)
    return C / np.sum(C)


def entropy_weights(matrix, *args, **kwargs):
    """ Calculate weights for given `matrix` using entropy method.

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

    m, n = matrix.shape
    # nmatrix = normalize_matrix(matrix, sum_normalization, None)
    nmatrix = sum_normalization(matrix)
    entropies = np.empty(n)
    # Iterate over all criteria
    for i, col in enumerate(nmatrix.T):
        if np.any(col == 0):
            entropies[i] = 0
        else:
            entropies[i] = -np.sum(col * np.log(col))
    entropies = entropies / np.log(m)

    E = 1 - entropies
    return E / np.sum(E)



def AHP_weights(features,n):
    pairs=np.ones((n,n))
    for i in range(len(features)):
        j=i+1
        while j<len(features):
            pairs[i, j] = float(
                input("Enter factor of "+features[i]+" vs "+features[j]+" : "))
            pairs[j,i]=1/pairs[i,j]
            j+=1
    v=np.zeros(n)
    for i in range(n):
        t=1
        for j in range(n):
            t=t*(pairs[i,j]) 
        v[i]=t**(1/n)
    return v/np.sum(v)       


# if __name__ == "__main__":
#     print(AHP_weights(["cost","loc","Rank"],3))

