import numpy as np
import math

bandwidth = 20*pow(10, 6)
transmissionPowerEndDevice = 0.5  # 0.5w
n0 = pow(10, -10)


def calculateLatency(Task, FogNode):

    #print("Task: ", Task)
    taskSize = Task[0]
    dist = FogNode[3]

    pathloss = 38.02+20*math.log10(dist)
    channelGain = pow(10, -1*(pathloss/10))
    uploadingRate = bandwidth * \
        math.log2(1+(transmissionPowerEndDevice*channelGain)/n0)
    uploadingTime = taskSize/uploadingRate

    computingRate = FogNode[1]
    computationTime = taskSize/computingRate

    outputSize = Task[2]
    transmissionPowerOfFog = FogNode[4]
    downloadingRate = bandwidth * \
        math.log2(1+(transmissionPowerOfFog*channelGain)/n0)
    downloadingTime = outputSize / downloadingRate

    return np.array([uploadingTime, computationTime, downloadingTime])


def calculateEnergy(Task, FogNode, latencyPerTask):
    uploadingTime = latencyPerTask[0]
    uploadingTransmissionEnergy = uploadingTime*transmissionPowerEndDevice

    computingTime = latencyPerTask[1]
    computationEnergy = computingTime*FogNode[0]

    downloadingTime = latencyPerTask[2]
    transmissionPowerOfFog = FogNode[4]
    downloadingEnergy = downloadingTime*transmissionPowerOfFog

    return np.array([uploadingTransmissionEnergy, computationEnergy, downloadingEnergy])


def matchTaskandFogNode(TaskOrdering, FogNodesOrdering):


    taskIter = 0
    totLatency = 0
    totEnergy = 0

    noOfFogNodes = len(FogNodesOrdering)
    noOfTasks = len(TaskOrdering)
    totParallel = 0
    for fogNode in FogNodesOrdering:
        totParallel += int(fogNode[2])
    # print("Tot Parallel: ", totParallel)
    print("================================")
    print("Total Parallel: ", totParallel, " total Tasks: ", noOfTasks)
    print("==================================================")
    for fogNode in FogNodesOrdering:
        parallelism = int(fogNode[2])
        while parallelism > 0 and taskIter < noOfTasks:
            Task = TaskOrdering[taskIter]
            # print("Task in Loop: ",Task)
            latencyPerTask = calculateLatency(Task, fogNode)
            totLatency = np.sum(latencyPerTask)
            energyPerTask = calculateEnergy(
                TaskOrdering[taskIter], fogNode, latencyPerTask)
            totEnergy += np.sum(energyPerTask)
            parallelism -= 1
            taskIter += 1

    return np.array([totLatency, totEnergy])
