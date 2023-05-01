import numpy as np
from methods.TOPSIS import TOPSIS
from methods.moora import MOORA
from methods.vikor import VIKOR
from helpers import rrankdata

topsis = TOPSIS()
moora = MOORA()
vikor=VIKOR()


def getFinalRankTable(DecisionMatrix, ranking, PrefValue):
    table = []
    for r, task in zip(ranking, DecisionMatrix):
        # print(r, " ", task)
        table.append([r,task])
    table=sorted(table)
    finalAlternativeTable = []
    for t in table:
        finalAlternativeTable.append(t[1])
    return finalAlternativeTable



def getOrderByTopsis(DecisionMatrix, Weight, types):
    PrefValue = topsis(np.array(DecisionMatrix),
                       np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable


def getOrderByMoora(DecisionMatrix, Weight, types):
    PrefValue = moora(np.array(DecisionMatrix),
                      np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable


def getOrderByVikor(DecisionMatrix, Weight, types):
    PrefValue = vikor(np.array(DecisionMatrix),
                      np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable


