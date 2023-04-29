from numpy import random
import numpy as np


class Fognode:
    def __init__(self, power, computingRate, parallelism, distance, transmissionPower):
        self.power = power
        self.computingRate = computingRate
        self.parallelism = parallelism
        self.distance = distance
        self.transmissionPower = transmissionPower

    def convertFogAttributesToList(self):
        list = []
        list.extend([self.power, self.computingRate,
                    self.parallelism, self.distance, self.transmissionPower])
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
    rateNormalDistributed = random.normal(5, 2, count)
    powerNormalDistributed = random.normal(1, 0.3, count)
    for i in range(count):
        dist = random.randint(10, 50)
        parallelism = random.randint(3, 8)
        transmissionPower = random.randint(10, 20)
        f = Fognode(
            powerNormalDistributed[i], rateNormalDistributed[i], parallelism, dist, transmissionPower)
        fogNodes.append(f)

    return fogNodes
