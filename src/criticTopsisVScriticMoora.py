import numpy as np
from numpy import random
from normalisations import minmax_normalization
from weights import critic_weights, entropy_weights
from Task import Task, generateTasks, convertTasksToMatrix
from Fognode import Fognode, convertFogNodesToMatrix, generateFogNodes
from methods.TOPSIS import TOPSIS
from methods.moora import MOORA
from helpers import rrankdata
from mcda_method_call import getOrderByTopsis, getOrderByMoora
from matchTaskandFogNode import matchTaskandFogNode
from plot import generatePlot


def main():
    noOfTasks = 10
    noOfFogNodes = 25
    fogNodes = generateFogNodes(noOfFogNodes)
    fogDecisionMatrix = convertFogNodesToMatrix(fogNodes)
    fogNodesCriticWeight = critic_weights(fogDecisionMatrix)
    FogNodesTopsisOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesCriticWeight, [-1, 1, 1, -1, -1])

    FogNodesMooraOrdering = getOrderByMoora(
        fogDecisionMatrix, fogNodesCriticWeight, [-1, 1, 1, -1, -1])
    taskCount = np.array([])
    topsisLatency = np.array([])
    topsisEnergy = np.array([])
    mooraLatency = np.array([])
    mooraEnergy = np.array([])
    for i in range(100):
        tasks = generateTasks(noOfTasks)
        tasksDecisionMatrix = convertTasksToMatrix(tasks)

        tasksCriticWeight = critic_weights(tasksDecisionMatrix)

        # print("No of Tasks: ", noOfTasks)

        topsisTaskOrdering = getOrderByTopsis(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])
        mooraTaskOrdering = getOrderByMoora(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])


        [totLatencyTopsis, totEnergyTopsis] = matchTaskandFogNode(
            topsisTaskOrdering, FogNodesTopsisOrdering)

        [totLatencyMoora, totEnergyMoora] = matchTaskandFogNode(
            mooraTaskOrdering, FogNodesMooraOrdering)

        print("totLatency: ", totLatencyTopsis, " == ", totLatencyMoora)
        print("totEnergy: ", totEnergyTopsis, " == ", totEnergyMoora)

        taskCount = np.append(taskCount, noOfTasks)
        topsisLatency = np.append(topsisLatency, totLatencyTopsis)
        topsisEnergy = np.append(topsisEnergy, totEnergyTopsis)

        mooraLatency = np.append(mooraLatency, totLatencyMoora)
        mooraEnergy = np.append(mooraEnergy, totEnergyMoora)

        noOfTasks = noOfTasks+10

    print("==============================")
    print("For Energy: ")
    print("Topsis ENergy: ", topsisEnergy)
    print("Moora Energy: ", mooraEnergy)
    print("==============================")
    print("==============================")
    print("For letency : ")
    print("Topsis Latency: ", topsisLatency)
    print("Moora Latency: ", mooraLatency)
    print("==============================")
    generatePlot(taskCount, topsisEnergy, mooraEnergy,
                 "Task Count", "Energy", "with TOPSIS", "with MOORA")

    
    generatePlot(taskCount, topsisLatency, mooraLatency,
                 "Task Count", "Latency", "with TOPSIS", "with MOORA")



if __name__ == "__main__":
    main()
