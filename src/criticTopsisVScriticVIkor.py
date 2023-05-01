import numpy as np
from numpy import random
from normalisations import minmax_normalization
from weights import critic_weights, entropy_weights
from Task import Task, generateTasks, convertTasksToMatrix
from Fognode import Fognode, convertFogNodesToMatrix, generateFogNodes
from methods.TOPSIS import TOPSIS
from methods.moora import MOORA
from helpers import rrankdata
from mcda_method_call import getOrderByTopsis, getOrderByMoora,getOrderByVikor
from matchTaskandFogNode import matchTaskandFogNode
from plot import generatePlot


def main():
    noOfTasks = 10
    noOfFogNodes = 50
    fogNodes = generateFogNodes(noOfFogNodes)
    fogDecisionMatrix = convertFogNodesToMatrix(fogNodes)
    fogNodesCriticWeight = critic_weights(fogDecisionMatrix)
    FogNodesTopsisOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesCriticWeight, [-1, 1, 1, -1, -1])

    FogNodesMooraOrdering = getOrderByVikor(
        fogDecisionMatrix, fogNodesCriticWeight, [-1, 1, 1, -1, -1])
    taskCount = np.array([])
    topsisLatency = np.array([])
    topsisEnergy = np.array([])
    vikorLatency = np.array([])
    vikorEnergy = np.array([])
    for i in range(75):
        tasks = generateTasks(noOfTasks)
        tasksDecisionMatrix = convertTasksToMatrix(tasks)

        tasksCriticWeight = critic_weights(tasksDecisionMatrix)

        topsisTaskOrdering = getOrderByTopsis(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])
        vikorTaskOrdering = getOrderByVikor(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])

        [totLatencyTopsis, totEnergyTopsis] = matchTaskandFogNode(
            topsisTaskOrdering, FogNodesTopsisOrdering)

        [totLatencyVikor, totEnergyVikor] = matchTaskandFogNode(
            vikorTaskOrdering, FogNodesMooraOrdering)

        print("totLatency: ", totLatencyTopsis, " == ", totLatencyVikor)
        print("totEnergy: ", totEnergyTopsis, " == ", totEnergyVikor)

        taskCount = np.append(taskCount, noOfTasks)
        topsisLatency = np.append(topsisLatency, totLatencyTopsis)
        topsisEnergy = np.append(topsisEnergy, totEnergyTopsis)

        vikorLatency = np.append(vikorLatency, totLatencyVikor)
        vikorEnergy = np.append(vikorEnergy, totEnergyVikor)

        noOfTasks = noOfTasks+10

    # print("==============================")
    # print("For Energy: ")
    # print("Topsis ENergy: ", topsisEnergy)
    # print("Vikor Energy: ", vikorEnergy)
    # print("==============================")
    # print("==============================")
    # print("For letency : ")
    # print("Topsis Latency: ", topsisLatency)
    # print("Vikor Latency: ", vikorLatency)
    # print("==============================")
    generatePlot(taskCount, topsisEnergy, vikorEnergy,
                 "Task Count", "Energy", "with TOPSIS", "with vikor")

    generatePlot(taskCount, topsisLatency, vikorLatency,
                 "Task Count", "Latency", "with TOPSIS", "with vikor")


if __name__ == "__main__":
    main()
