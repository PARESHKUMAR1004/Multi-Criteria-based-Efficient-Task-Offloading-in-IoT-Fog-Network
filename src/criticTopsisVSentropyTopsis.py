import numpy as np

from weights import critic_weights, entropy_weights
from Task import generateTasks, convertTasksToMatrix
from Fognode import convertFogNodesToMatrix, generateFogNodes
from mcda_method_call import getOrderByTopsis
from matchTaskandFogNode import matchTaskandFogNode
from plot import generatePlot


def main():
    noOfTasks = 10
    noOfFogNodes = 25
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
    for i in range(35):
        tasks = generateTasks(noOfTasks)
        tasksDecisionMatrix = convertTasksToMatrix(tasks)
        tasksCriticWeight = critic_weights(tasksDecisionMatrix)
        tasksEntropyWeight = entropy_weights(np.array(tasksDecisionMatrix))
        print("No of Tasks: ", noOfTasks)

        finalEntropyTaskOrdering = getOrderByTopsis(
            tasksDecisionMatrix, tasksEntropyWeight, [1, -1, 1])
        finalCriticTaskOrdering = getOrderByTopsis(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])

        # print("Entropy Ordering Task: ", finalEntropyTaskOrdering)
        # print("Critic Ordering Task: ", finalCriticTaskOrdering)

        # print("No of Tasks: ", noOfTasks)
        # print("Final Task Ordering: ", finalTaskOrdering)
        [totLatencyCritic, totEnergyCritic] = matchTaskandFogNode(
            finalCriticTaskOrdering, finalFogNodesOrdering)

        [totLatencyEntropy, totEnergyEntropy] = matchTaskandFogNode(
            finalEntropyTaskOrdering, finalFogNodesOrdering)

        print("totLatency: ", totLatencyCritic, " == ", totLatencyEntropy)
        print("totEnergy: ", totEnergyCritic, " == ", totEnergyEntropy)

        taskCount = np.append(taskCount, noOfTasks)
        criticLatency = np.append(criticLatency, totLatencyCritic)
        criticEnergy = np.append(criticEnergy, totEnergyCritic)

        entropyLatency = np.append(entropyLatency, totLatencyEntropy)
        entropyEnergy = np.append(entropyEnergy, totEnergyEntropy)

        noOfTasks = noOfTasks+10

    generatePlot(taskCount, criticEnergy, entropyEnergy,
                 "Task Count", "Energy", "with Critic", "with Entropy")

    generatePlot(taskCount, criticLatency, entropyLatency,
                 "Task Count", "Latency", "with Critic", "with Entropy")


if __name__ == "__main__":
    main()
