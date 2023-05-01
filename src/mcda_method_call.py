import numpy as np
from methods.TOPSIS import TOPSIS
from methods.moora import MOORA
from helpers import rrankdata

topsis = TOPSIS()
moora = MOORA()


def getOrderByTopsis(DecisionMatrix, Weight, types):
    PrefValue = topsis(np.array(DecisionMatrix),
                       np.array(Weight), np.array(types))
    # print("Pref Value: ", PrefValue)
    ranking = rrankdata(PrefValue)
    noOfAlternatives = len(DecisionMatrix)
    finalAlternativeTable = np.zeros(noOfAlternatives, np.ndarray)
    iter = 0
    Dict = {}
    for r, p in zip(ranking, PrefValue):
        task = np.array(DecisionMatrix[iter])
        # print("R= ", r, " pref: ", p, " task: ", task)
        if (r-int(r)) > 0:
            print("R: ", r)
            if Dict.get(r) == None:
                Dict[r] = []
                Dict[r].append(task)
            else:
                Dict[r].append(task)
        finalAlternativeTable[int(r)-1] = task
        iter += 1

    for k, v in Dict.items():
        n = len(v)
        firstValue = k-(n-1)/2
        print("K: ", k, " V: ", v, " FirstValue: ", firstValue)
        i = 0
        while i < n:
            finalAlternativeTable[int(firstValue)] = v[i]
            i += 1
            firstValue += 1
    return finalAlternativeTable


def getOrderByMoora(DecisionMatrix, Weight, types):
    PrefValue = moora(np.array(DecisionMatrix),
                      np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    noOfAlternatives = len(DecisionMatrix)
    finalAlternativeTable = np.zeros(noOfAlternatives, np.ndarray)
    iter = 0
    Dict = {}
    for r, p in zip(ranking, PrefValue):
        task = np.array(DecisionMatrix[iter])
        if (r-int(r)) > 0:
            print("R: ", r)
            if Dict.get(r) == None:
                Dict[r] = []
                Dict[r].append(task)
            else:
                Dict[r].append(task)
        finalAlternativeTable[int(r)-1] = task
        iter += 1

    for k, v in Dict.items():
        n = len(v)
        firstValue = k-(n-1)/2
        i = 0
        while i < n:
            finalAlternativeTable[int(firstValue)] = v[i]
            i += 1
            firstValue += 1
    return finalAlternativeTable
