from abc import ABC
import numpy as np
from collections import Counter


def rankdata(a, reverse=False):
    """
    Assign ranks to data in vector `a`.

    Ranks begin at 1. Tied elements get average rank (see Examples below).

    Ranking starts from smaller values, e.g. the smaller element get
    the first position. The `reverse` argument reverse posisions, e.g.
    the largest element get first position.

    Parameters
    ----------
    a : iterable
        The array of values to be ranked.

    reverse : bool, optional
        If True, larger elements get first posisions in ranking.
        If False, smaller elements get first positions in ranking.

    Returns
    -------
    ndarray
        An array of rank scores for the input data.

    Examples
    --------
    >>> from pymcdm.helpers import rankdata
    >>> rankdata([0, 3, 2, 5])
    array([1, 3, 2, 4])
    >>> rankdata([0, 3, 2, 5], reverse=True)
    array([4, 2, 3, 1])
    >>> rankdata([0, 3, 2, 3])
    array([1. , 3.5, 2. , 3.5])
    >>> rankdata([0, 3, 2, 3], reverse=True)
    array([4. , 1.5, 3. , 1.5])
    """
    c = Counter(a)
    rv = {}
    i = 1
    for k in sorted(c.keys(), reverse=reverse):
        if c[k] == 1:
            rv[k] = i
            i += 1
        else:
            v = c[k]
            rv[k] = (2*i + v - 1)/2
            i += v
    return np.array([rv[k] for k in a], dtype='float')


class MCDA_method(ABC):
    reverse_ranking = True

    def __call__(self, matrix, weights, types, *args, **kwargs):
        """Rank alternatives from decision matrix `matrix`, with criteria weights `weights` and criteria types `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Criteria weights. Sum of the weights should be 1. (e.g. sum(weights) == 1)

            types : ndarray
                Array with definitions of criteria types:
                1 if criteria is profit and -1 if criteria is cost for each criteria in `matrix`.

            *args: is necessary for methods which reqiure some additional data.

            **kwargs: is necessary for methods which reqiure some additional data.
        """
        pass

    @staticmethod
    def _validate_input_data(matrix, weights, types):
        if matrix.shape[1] != weights.shape[0] and weights.shape[0] != len(types):
            raise ValueError(
                f'Number of criteria should be same as number of weights and number of types')

    def rank(self, a):
        return rankdata(a, reverse=self.reverse_ranking)
