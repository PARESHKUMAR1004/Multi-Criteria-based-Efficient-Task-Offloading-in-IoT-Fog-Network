import numpy as np
import math

bandwidth = 10*pow(2, 20)#10 Mbits
transmissionPowerEndDevice = 0.5  # 0.5w
n0 = pow(10, -10)


def calculateLatency(Task, FogNode):

    #print("Task: ", Task)
    taskSize = Task[0]
    dist = FogNode[2]

    pathloss = 38.02+20*math.log10(dist)
    channelGain = pow(10, -1*(pathloss/10))
    uploadingRate = bandwidth * math.log2(1+(transmissionPowerEndDevice*channelGain)/n0)
    uploadingTime = taskSize/uploadingRate

    computingRate = FogNode[1]
    
    computationTime = taskSize/computingRate

    outputSize = Task[2]
    transmissionPowerOfFog = 1
    downloadingRate = bandwidth * math.log2(1+(transmissionPowerOfFog*channelGain)/n0)
    downloadingTime = outputSize / downloadingRate
    # print(uploadingTime," ",computationTime," ",downloadingTime)
    #print("Uploading Time: ",uploadingTime,"Computation Time: ",computationTime,"Downloading Time: ",downloadingTime)
    #print("Task Deadline: ", Task[1], " Completion Time: ", np.sum(np.array([uploadingTime, computationTime, downloadingTime])))
    return np.array([uploadingTime, computationTime, downloadingTime])


def calculateEnergy(Task, FogNode, latencyPerTask):
    uploadingTime = latencyPerTask[0]
    uploadingTransmissionEnergy = uploadingTime*transmissionPowerEndDevice

    computingTime = latencyPerTask[1]
    computationEnergy = computingTime*FogNode[0]

    downloadingTime = latencyPerTask[2]
    transmissionPowerOfFog = 1
    downloadingEnergy = downloadingTime*transmissionPowerOfFog

    # print(uploadingTransmissionEnergy," ",computationEnergy," ",downloadingEnergy)
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

    noOfOutage=0
    for fogNode in FogNodesOrdering:
        parallelism = int(fogNode[3])
        while parallelism > 0 and taskIter < noOfTasks:
            Task = TaskOrdering[taskIter]
            taskDeadline=Task[1]
            latencyPerTask = calculateLatency(Task, fogNode)
            latency = np.sum(latencyPerTask)
            if latency>taskDeadline:
                noOfOutage+=1

        
            totLatency += latency
            energyPerTask = calculateEnergy(Task, fogNode, latencyPerTask)
            totEnergy += np.sum(energyPerTask)
            parallelism -= 1
            taskIter += 1

    return np.array([totLatency, totEnergy,noOfOutage])
