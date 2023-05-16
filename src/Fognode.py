from numpy import random
import numpy as np


class Fognode:
    def __init__(self, power, computingRate,VRU, distance, transmissionPower):
        self.power = power
        self.computingRate = computingRate
        self.VRU = VRU
        self.ratePerVRU=self.computingRate/self.VRU
        self.distance = distance
        self.transmissionPower = 1

    def convertFogAttributesToList(self):
        list = []
        list.extend([self.power, self.ratePerVRU, self.distance,self.VRU])
        return list

    def showFogNodeAttributes(self):
        print("Fog Node Power: ", self.power)
        print("Fog Node Computing Rate: ", self.computingRate)
        print("Fog Node Parallelism: ", self.parallelism)
        print("Fog Node Distance: ", self.distance)


def convertFogNodesToMatrix(fogNodes):
    DecisionMatrix = []
    for i in range(len(fogNodes)):
        DecisionMatrix.append(fogNodes[i].convertFogAttributesToList())

    return DecisionMatrix


def generateFogNodes(count):
    fogNodes = []
    # Got from fair task offloading paper
    rateNormalDistributed = []
    powerNormalDistributed=[]
    for i in range(count):
        rateNormalDistributed.append(random.uniform(0.5,1.5)*pow(2,30))
        powerNormalDistributed.append(random.uniform(0.5,1.5))
        
    
    print("Rate: ", rateNormalDistributed)
    
    print("Power: ", powerNormalDistributed)
    for i in range(count):
        dist = random.randint(50, 500)  # Got from chitta Sir's Paper
        parallelism = random.randint(100, 200)
        # Every fog Node has it's own transmission Power
        transmissionPower = random.randint(1, 10)
        f = Fognode(
            powerNormalDistributed[i], rateNormalDistributed[i], parallelism, dist, transmissionPower)
        fogNodes.append(f)

    return fogNodes
