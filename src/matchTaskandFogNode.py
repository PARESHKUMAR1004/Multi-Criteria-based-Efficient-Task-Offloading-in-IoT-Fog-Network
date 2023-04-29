import numpy as np
import math

bandwidth = 100
k = 25
transmissionPowerEndDevice = 24


def calculateLatency(Task, FogNode):
    # print("Latency")
    # print("Task: ", Task)
    # print("FogNode ", FogNode)

    taskSize = Task[0]
    dist = FogNode[3]
    uploadingTime = taskSize/(bandwidth*math.log2(1+(k*(23+math.log2(dist)))))
    computingRate = FogNode[1]
    computationTime = taskSize/computingRate

    outputSize = Task[2]
    downloadingTime = outputSize / \
        (bandwidth*math.log2(1+(k*(23+math.log2(dist)))))
    return np.array([uploadingTime, computationTime, downloadingTime])


def calculateEnergy(Task, FogNode, latencyPerTask):
    uploadingTime = latencyPerTask[0]
    uploadingTransmissionEnergy = uploadingTime*transmissionPowerEndDevice

    computingTime = latencyPerTask[1]
    computationEnergy = computingTime*FogNode[0]

    downloadingTime = latencyPerTask[2]
    downloadingEnergy = downloadingTime*FogNode[4]

    return np.array([uploadingTransmissionEnergy, computationEnergy, downloadingEnergy])


def matchTaskandFogNode(TaskOrdering, FogNodesOrdering):

    # print("Task Ordering: ", TaskOrdering)
    # print("Fog Node Ordering: ", FogNodesOrdering)
    # totLatency=0
    taskIter = 0
    totLatency = 0
    totEnergy = 0

    noOfFogNodes = len(FogNodesOrdering)
    noOfTasks = len(TaskOrdering)
    for fogNode in FogNodesOrdering:
        parallelism = int(fogNode[2])
        while parallelism > 0 and taskIter < noOfTasks:
            latencyPerTask = calculateLatency(TaskOrdering[taskIter], fogNode)
            totLatency = np.sum(latencyPerTask)
            energyPerTask = calculateEnergy(
                TaskOrdering[taskIter], fogNode, latencyPerTask)
            totEnergy += np.sum(energyPerTask)
            parallelism -= 1
            taskIter += 1

    return np.array([totLatency, totEnergy])
