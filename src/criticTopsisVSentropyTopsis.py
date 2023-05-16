import numpy as np

from weights import critic_weights, entropy_weights
from Task import generateTasks, convertTasksToMatrix
from Fognode import convertFogNodesToMatrix, generateFogNodes
from mcda_method_call import getOrderByTopsis
from matchTaskandFogNode import matchTaskandFogNode
from plot import generatePlot


def main():
    noOfTasks = 200
    noOfFogNodes = 10
    fogNodes = generateFogNodes(noOfFogNodes)
    fogNodesType = [-1, 1, 1, -1, -1]
    fogDecisionMatrix = convertFogNodesToMatrix(fogNodes)
    fogNodesCriticWeight = critic_weights(fogDecisionMatrix)
    finalFogNodesOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesCriticWeight, fogNodesType)

    taskCount = np.array([])
    criticLatency = np.array([])
    criticEnergy = np.array([])
    entropyLatency = np.array([])
    entropyEnergy = np.array([])
    for i in range(5):
        totLatencyCritic=0
        totEnergyCritic=0
        totLatencyEntropy=0
        totEnergyEntropy=0


        for iter in range(10):
            tasks = generateTasks(noOfTasks)
            tasksDecisionMatrix = convertTasksToMatrix(tasks)
            tasksCriticWeight = critic_weights(tasksDecisionMatrix)
            tasksEntropyWeight = entropy_weights(np.array(tasksDecisionMatrix))
            # print("No of Tasks: ", noOfTasks)

            finalEntropyTaskOrdering = getOrderByTopsis(tasksDecisionMatrix, tasksEntropyWeight, [1, -1, 1])
            finalCriticTaskOrdering = getOrderByTopsis(tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])

        
            [LatencyCritic, EnergyCritic] = matchTaskandFogNode(finalCriticTaskOrdering, finalFogNodesOrdering)
            [LatencyEntropy, EnergyEntropy] = matchTaskandFogNode(finalEntropyTaskOrdering, finalFogNodesOrdering)
            totLatencyCritic+=LatencyCritic
            totEnergyCritic+=EnergyCritic
            totLatencyEntropy+=LatencyEntropy
            totEnergyEntropy+=EnergyEntropy
        

        taskCount = np.append(taskCount, noOfTasks)
        criticLatency = np.append(criticLatency, totLatencyCritic/10)
        criticEnergy = np.append(criticEnergy, totEnergyCritic/10)

        entropyLatency = np.append(entropyLatency, totLatencyEntropy/10)
        entropyEnergy = np.append(entropyEnergy, totEnergyEntropy/10)

        noOfTasks = noOfTasks+200
    
    

    generatePlot(taskCount, criticEnergy, entropyEnergy,
                 "Task Count", "Energy", "with Critic", "with Entropy")

    generatePlot(taskCount, criticLatency, entropyLatency,
                 "Task Count", "Latency", "with Critic", "with Entropy")


if __name__ == "__main__":
    main()
