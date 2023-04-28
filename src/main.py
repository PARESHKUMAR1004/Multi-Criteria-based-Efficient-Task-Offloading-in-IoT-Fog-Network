import numpy as np
from numpy import random
from normalisations import minmax_normalization
from weights import critic_weights
from Task import Task, generateTasks, convertTasksToMatrix
from Fognode import Fognode, convertFogNodesToMatrix, generateFogNodes


task1 = Task(3000, 45, 200)
task1.showTaskAttributes()


m = convertFogNodesToMatrix(generateFogNodes(5))
print(m)


def main():
    print("Hii")
    noOfTasks = 10
    noOfFogNodes = 25
    fogNodes = generateFogNodes(noOfFogNodes)
    fogDecisionMatrix = convertFogNodesToMatrix(fogNodes)
    for i in range(10):
        tasks = generateTasks(noOfTasks)
        tasksDecisionMatrix = convertTasksToMatrix(tasks)
        tasksCriticWeight = critic_weights(tasksDecisionMatrix)
        fogNodesCriticWeight = critic_weights(fogDecisionMatrix)
        noOfTasks = noOfTasks+10
        print("Hiii")
        print(tasksCriticWeight)


if __name__ == "__main__":
    main()
