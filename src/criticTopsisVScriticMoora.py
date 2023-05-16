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
    noOfTasks = 200
    noOfFogNodes = 10
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
    for i in range(5):
        totLatencyTopsis=0
        totEnergyTopsis=0
        totLatencyMoora=0
        totEnergyMoora=0


        for iter in range(10):
            tasks = generateTasks(noOfTasks)
            tasksDecisionMatrix = convertTasksToMatrix(tasks)

            tasksCriticWeight = critic_weights(tasksDecisionMatrix)

        

            topsisTaskOrdering = getOrderByTopsis(tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])
            mooraTaskOrdering = getOrderByMoora(tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])


            [LatencyTopsis, EnergyTopsis] = matchTaskandFogNode(topsisTaskOrdering, FogNodesTopsisOrdering)
            [LatencyMoora, EnergyMoora] = matchTaskandFogNode(mooraTaskOrdering, FogNodesMooraOrdering)
            totLatencyTopsis+=LatencyTopsis
            totEnergyTopsis+=EnergyTopsis
            totLatencyMoora+=LatencyMoora
            totEnergyMoora+=EnergyMoora

        # print("totLatency: ", totLatencyTopsis, " == ", totLatencyMoora)
        # print("totEnergy: ", totEnergyTopsis, " == ", totEnergyMoora)

        taskCount = np.append(taskCount, noOfTasks)
        topsisLatency = np.append(topsisLatency, totLatencyTopsis/10)
        topsisEnergy = np.append(topsisEnergy, totEnergyTopsis/10)

        mooraLatency = np.append(mooraLatency, totLatencyMoora/10)
        mooraEnergy = np.append(mooraEnergy, totEnergyMoora/10)

        noOfTasks = noOfTasks+200

    for j in range(len(topsisEnergy)):
        print("Energy: ",topsisEnergy[j]," ",mooraEnergy[j])
        print("Latency: ",topsisLatency[j]," ",mooraLatency[j])
    generatePlot(taskCount, topsisEnergy, mooraEnergy,
                 "Task Count", "Energy", "with TOPSIS", "with MOORA")

    
    generatePlot(taskCount, topsisLatency, mooraLatency,
                 "Task Count", "Latency", "with TOPSIS", "with MOORA")



if __name__ == "__main__":
    main()
