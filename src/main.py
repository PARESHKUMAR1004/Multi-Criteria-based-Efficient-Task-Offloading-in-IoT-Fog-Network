import numpy as np
from numpy import random
from normalisations import minmax_normalization
from weights import critic_weights
from Task import Task, generateTasks, convertTasksToMatrix
from Fognode import Fognode, convertFogNodesToMatrix, generateFogNodes
# from methods import TOPSIS
from methods.TOPSIS import TOPSIS
from helpers import rrankdata
from mcda_method_call import getOrderByTopsis, getOrderByMoora
from matchTaskandFogNode import matchTaskandFogNode


def main():
    print("Hii")
    noOfTasks = 10
    noOfFogNodes = 25
    fogNodes = generateFogNodes(noOfFogNodes)
    fogDecisionMatrix = convertFogNodesToMatrix(fogNodes)
    fogNodesCriticWeight = critic_weights(fogDecisionMatrix)
    finalFogNodesOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesCriticWeight, [-1, 1, 1, -1, -1])
    # print("Fog Nodes Ordering: ")
    # print(finalFogNodesOrdering)
    for i in range(10):
        print("No of Tasks: ",noOfTasks)
        tasks = generateTasks(noOfTasks)
        tasksDecisionMatrix = convertTasksToMatrix(tasks)
        tasksCriticWeight = critic_weights(tasksDecisionMatrix)
        finalTaskOrdering = getOrderByTopsis(
            tasksDecisionMatrix, tasksCriticWeight, [1, -1, 1])

        # print("No of Tasks: ", noOfTasks)
        # print("Final Task Ordering: ", finalTaskOrdering)
        [totLatency, totEnergy] = matchTaskandFogNode(
            finalTaskOrdering, finalFogNodesOrdering)
        print("totLatency: ", totLatency)
        print("totEnergy: ", totEnergy)
        noOfTasks = noOfTasks+10


if __name__ == "__main__":
    main()
