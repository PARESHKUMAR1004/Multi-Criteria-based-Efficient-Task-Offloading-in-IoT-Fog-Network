import numpy as np
from weights import critic_weights
from Task import generateTasks, convertTasksToMatrix
from Fognode import convertFogNodesToMatrix, generateFogNodes
from mcda_method_call import getOrderByTopsis, getOrderByVikor, getOrderByMarcos
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

    FogNodesMarcosOrdering = getOrderByMarcos(
        fogDecisionMatrix, fogNodesCriticWeight, [-1, 1, 1, -1, -1])
    taskCount = np.array([])
    topsisLatency = np.array([])
    topsisEnergy = np.array([])
    marcosLatency = np.array([])
    marcosEnergy = np.array([])
    for i in range(5):
        totLatencyTopsis = 0
        totEnergyTopsis = 0
        totLatencyMarcos = 0
        totEnergyMarcos = 0

        for iter in range(10):
            tasks = generateTasks(noOfTasks)
            tasksDecisionMatrix = convertTasksToMatrix(tasks)

            tasksCriticWeight = critic_weights(tasksDecisionMatrix)

            topsisTaskOrdering = getOrderByTopsis(
                tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])
            marcosTaskOrdering = getOrderByMarcos(
                tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])

            [LatencyTopsis, EnergyTopsis] = matchTaskandFogNode(
                topsisTaskOrdering, FogNodesTopsisOrdering)

            [LatencyMarcos, EnergyMarcos] = matchTaskandFogNode(
                marcosTaskOrdering, FogNodesMarcosOrdering)
            totLatencyTopsis += LatencyTopsis
            totEnergyTopsis += EnergyTopsis
            totLatencyMarcos += LatencyMarcos
            totEnergyMarcos += EnergyMarcos

        taskCount = np.append(taskCount, noOfTasks)
        topsisLatency = np.append(topsisLatency, totLatencyTopsis/10)
        topsisEnergy = np.append(topsisEnergy, totEnergyTopsis/10)

        marcosLatency = np.append(marcosLatency, totLatencyMarcos/10)
        marcosEnergy = np.append(marcosEnergy, totEnergyMarcos/10)

        noOfTasks = noOfTasks+200

    for i in range(len(topsisEnergy)):
        print("Energy: ",topsisEnergy[i]," ",marcosEnergy[i]," Diff: ",topsisEnergy[i]-marcosEnergy[i])
        print("Latency: ",topsisLatency[i]," ",marcosLatency[i]," Diff: ",topsisLatency[i]-marcosLatency[i])
    
    generatePlot(taskCount, topsisEnergy, marcosEnergy,
                 "Task Count", "Energy", "with TOPSIS", "with MARCOS")

    generatePlot(taskCount, topsisLatency, marcosLatency,
                 "Task Count", "Latency", "with TOPSIS", "with MARCOS")


if __name__ == "__main__":
    main()
