import numpy as np
from methods.TOPSIS import TOPSIS
from methods.moora import MOORA
from methods.vikor import VIKOR
from methods.marcos import MARCOS
from helpers import rrankdata,rankdata






def getFinalRankTable(DecisionMatrix, ranking, PrefValue):
    # DecisionMatrix=list(DecisionMatrix)
    table = []
    for r, task in zip(ranking, DecisionMatrix):
        table.append([r,list(task)])

    # print("Table Here: ",table)
    # table(np.array(table))
    # table=table[table[:,0].argsort()]
    table=sorted(table)
    finalAlternativeTable = []
    for t in table:
        finalAlternativeTable.append(t[1])
    return finalAlternativeTable



def getOrderByTopsis(DecisionMatrix, Weight, types,columns):
    topsis = TOPSIS()
    # print("Decision Matrix: ",DecisionMatrix)
    # print("Changed Matrix: ",DecisionMatrix[:,:columns])
    PrefValue = topsis(DecisionMatrix[:,:columns],
                       np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable


def getOrderByMoora(DecisionMatrix, Weight, types,columns):
    moora = MOORA()
    PrefValue = moora(DecisionMatrix[:, :columns],
                      np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable


def getOrderByVikor(DecisionMatrix, Weight, types,columns):
    vikor = VIKOR()
    PrefValue = vikor(DecisionMatrix[:, :columns],
                      np.array(Weight), np.array(types))
    ranking = rankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable


def getOrderByMarcos(DecisionMatrix, Weight, types,columns):
    marcos=MARCOS()
    PrefValue = marcos(DecisionMatrix[:, :columns],
                      np.array(Weight), np.array(types))
    ranking = rrankdata(PrefValue)
    finalAlternativeTable = getFinalRankTable(
        DecisionMatrix, ranking, PrefValue)
    return finalAlternativeTable

