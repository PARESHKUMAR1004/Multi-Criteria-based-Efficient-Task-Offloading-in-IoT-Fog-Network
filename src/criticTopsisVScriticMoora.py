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
    for i in range(35):
        tasks = generateTasks(noOfTasks)
        tasksDecisionMatrix = convertTasksToMatrix(tasks)
        # for task in tasksDecisionMatrix:
        #     if isinstanc
        print("**************************")
        print("Tasks Decision Matrix Here: ")
        for task in tasksDecisionMatrix:
            print("Task: ",task)
        print("***************************")
        tasksCriticWeight = critic_weights(tasksDecisionMatrix)

        print("No of Tasks: ", noOfTasks)

        topsisTaskOrdering = getOrderByTopsis(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])
        mooraTaskOrdering = getOrderByMoora(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])
        print("Topsis Task Ordering: ")
        for task in topsisTaskOrdering:
            print("Task: ",task)
        print("================================")
        # print("Moora Task Ordering: ", mooraTaskOrdering)

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

    generatePlot(taskCount, topsisEnergy, mooraEnergy,
                 "Task Count", "Energy", "with TOPSIS", "with MOORA")

    generatePlot(taskCount, topsisLatency, mooraLatency,
                 "Task Count", "Latency", "with TOPSIS", "with MOORA")


if __name__ == "__main__":
    main()
