import numpy as np

from weights import critic_weights,entropy_weights
from Task import generateTasks, convertTasksToMatrix
from Fognode import convertFogNodesToMatrix, generateFogNodes
from mcda_method_call import getOrderByTopsis, getOrderByMarcos,getOrderByMoora,getOrderByVikor
from matchTaskandFogNode import matchTaskandFogNode
from plot import generatePlot


def main():
    noOfTasks = 200
    noOfFogNodes = 10
    fogNodes = generateFogNodes(noOfFogNodes)
    for node in fogNodes:
        print("VRU: ",node.VRU)
    fogNodesType = [-1, 1,-1]
    fogDecisionMatrix = np.array(convertFogNodesToMatrix(fogNodes))

    fogNodesCriticWeight = entropy_weights(fogDecisionMatrix[:,:-1])
    print("Fog Node Entropy Weight: ",fogNodesCriticWeight)
    fogNodesTopsisOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesCriticWeight, fogNodesType,len(fogDecisionMatrix[0])-1)
    fogNodesMarcosOrdering=getOrderByMarcos(fogDecisionMatrix,fogNodesCriticWeight,fogNodesType,len(fogDecisionMatrix[0])-1)
    fogNodesMooraOrdering = getOrderByMoora(
        fogDecisionMatrix, fogNodesCriticWeight, fogNodesType,len(fogDecisionMatrix[0])-1)
    fogNodesVikorOrdering = getOrderByVikor(
        fogDecisionMatrix, fogNodesCriticWeight, fogNodesType,len(fogDecisionMatrix[0])-1)
    print("================================")
    print("Topsis Ordering: ")
    for node in fogNodesTopsisOrdering:
        print(node)
    print("================================")

    print("Marcos Ordering: ")
    for node in fogNodesMarcosOrdering:
        print(node)
    
    print("================================")

    print("Moora Ordering: ")
    for node in fogNodesMooraOrdering:
        print(node)
    print("================================")

    print("Vikor Ordering: ")
    for node in fogNodesVikorOrdering:
        print(node)


    taskCount = np.array([])
    topsisLatency = np.array([])
    topsisEnergy = np.array([])
    marcosLatency = np.array([])
    marcosEnergy = np.array([])
    mooraLatency = np.array([])
    mooraEnergy = np.array([])
    vikorLatency = np.array([])
    vikorEnergy = np.array([])
    for i in range(5):
        totLatencyTopsis = 0
        totEnergyTopsis = 0
        totLatencyMarcos = 0
        totEnergyMarcos = 0
        totLatencyMoora = 0
        totEnergyMoora = 0
        totLatencyVikor = 0
        totEnergyVikor = 0
        totalIter=10
        for iter in range(totalIter):
            tasks = generateTasks(noOfTasks)
            tasksDecisionMatrix = np.array(convertTasksToMatrix(tasks))
            columns=len(tasksDecisionMatrix[0])
            tasksCriticWeight = entropy_weights(tasksDecisionMatrix)
        
            topsisTaskOrdering = getOrderByTopsis(
                tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1],columns)
            marcosTaskOrdering = getOrderByMarcos(
                tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1],columns)
            mooraTaskOrdering = getOrderByMoora(
                tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1],columns)
            vikorTaskOrdering = getOrderByVikor(
                tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1],columns)

            [latencyTopsis, energyTopsis] = matchTaskandFogNode(topsisTaskOrdering,fogNodesTopsisOrdering)
            [latencyMarcos, energyMarcos] = matchTaskandFogNode(
                marcosTaskOrdering, fogNodesMarcosOrdering)
            [latencyMoora, energyMoora] = matchTaskandFogNode(
                mooraTaskOrdering, fogNodesMooraOrdering)
            [latencyVikor, energyVikor] = matchTaskandFogNode(
                vikorTaskOrdering, fogNodesVikorOrdering)

            totLatencyTopsis += latencyTopsis
            totEnergyTopsis += energyTopsis
            totLatencyMarcos += latencyMarcos
            totEnergyMarcos += energyMarcos
            totLatencyMoora += latencyMoora
            totEnergyMoora += energyMoora
            totLatencyVikor += latencyVikor
            totEnergyVikor += energyVikor

        taskCount = np.append(taskCount, noOfTasks)
        topsisLatency = np.append(topsisLatency, totLatencyTopsis/totalIter)
        topsisEnergy = np.append(topsisEnergy, totEnergyTopsis/totalIter)

        marcosLatency = np.append(marcosLatency, totLatencyMarcos/totalIter)
        marcosEnergy = np.append(marcosEnergy, totEnergyMarcos/totalIter)

        mooraLatency = np.append(mooraLatency, totLatencyMoora/totalIter)
        mooraEnergy = np.append(mooraEnergy, totEnergyMoora/totalIter)
        
        vikorLatency = np.append(vikorLatency, totLatencyVikor/totalIter)
        vikorEnergy = np.append(vikorEnergy, totEnergyVikor/totalIter)




        noOfTasks = noOfTasks+200

    generatePlot(taskCount, topsisEnergy, marcosEnergy,mooraEnergy,vikorEnergy,
                 "Task Count", "Energy", "with TOPSIS", "with MARCOS", "with MOORA","With VIKOR")

    generatePlot(taskCount, topsisLatency, marcosLatency,mooraLatency,vikorLatency,
                 "Task Count", "Latency", "with TOPSIS", "with MARCOS", "with MOORA", "With VIKOR")


if __name__ == "__main__":
    main()
