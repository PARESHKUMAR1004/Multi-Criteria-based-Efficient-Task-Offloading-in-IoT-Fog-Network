import numpy as np

from weights import critic_weights, entropy_weights, AHP_weights
from Task import generateTasks, convertTasksToMatrix
from Fognode import convertFogNodesToMatrix, generateFogNodes
from mcda_method_call import getOrderByTopsis, getOrderByMarcos, getOrderByMoora, getOrderByVikor
from matchTaskandFogNode import matchTaskandFogNode
from plot import generatePlot, generatePlot2


def main():
    noOfTasks = 200
    noOfFogNodes = 20
    fogNodes = generateFogNodes(noOfFogNodes)
    for node in fogNodes:
        print("VRU: ", node.VRU)
    fogNodesType = [-1, 1, -1]
    fogDecisionMatrix = np.array(convertFogNodesToMatrix(fogNodes))

    fogNodesEntropyWeight = entropy_weights(fogDecisionMatrix[:, :-1])
    fogNodesAHPWeight = AHP_weights(["Power", "Rate per VRU", "Distance"], 3)
    print("Fog Node Entropy Weight: ", fogNodesEntropyWeight)
    print("Fog Node AHP Weight: ", fogNodesAHPWeight)
    fogEntropyTopsisOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesEntropyWeight, fogNodesType, len(fogDecisionMatrix[0])-1)
    fogEntropyMarcosOrdering = getOrderByMarcos(
        fogDecisionMatrix, fogNodesEntropyWeight, fogNodesType, len(fogDecisionMatrix[0])-1)
    fogEntropyMooraOrdering = getOrderByMoora(
        fogDecisionMatrix, fogNodesEntropyWeight, fogNodesType, len(fogDecisionMatrix[0])-1)
    fogEntropyVikorOrdering = getOrderByVikor(
        fogDecisionMatrix, fogNodesEntropyWeight, fogNodesType, len(fogDecisionMatrix[0])-1)

    fogAHPTopsisOrdering = getOrderByTopsis(
        fogDecisionMatrix, fogNodesAHPWeight, fogNodesType, len(fogDecisionMatrix[0])-1)
    fogAHPMarcosOrdering = getOrderByMarcos(
        fogDecisionMatrix, fogNodesAHPWeight, fogNodesType, len(fogDecisionMatrix[0])-1)
    fogAHPMooraOrdering = getOrderByMoora(
        fogDecisionMatrix, fogNodesAHPWeight, fogNodesType, len(fogDecisionMatrix[0])-1)
    fogAHPVikorOrdering = getOrderByVikor(
        fogDecisionMatrix, fogNodesAHPWeight, fogNodesType, len(fogDecisionMatrix[0])-1)

    taskCount = np.array([])
    tasksAHPWeight = AHP_weights(
        ["Task Size", "Task Deadline", "Output Size"], 3)
    topsisEntropyLatency = np.array([])
    topsisEntropyEnergy = np.array([])
    marcosEntropyLatency = np.array([])
    marcosEntropyEnergy = np.array([])
    mooraEntropyLatency = np.array([])
    mooraEntropyEnergy = np.array([])
    vikorEntropyLatency = np.array([])
    vikorEntropyEnergy = np.array([])

    topsisAHPLatency = np.array([])
    topsisAHPEnergy = np.array([])
    marcosAHPLatency = np.array([])
    marcosAHPEnergy = np.array([])
    mooraAHPLatency = np.array([])
    mooraAHPEnergy = np.array([])
    vikorAHPLatency = np.array([])
    vikorAHPEnergy = np.array([])

    topsisEntropyOutage=np.array([])
    marcosEntropyOutage = np.array([])
    mooraEntropyOutage = np.array([])
    vikorEntropyOutage = np.array([])

    topsisAHPOutage = np.array([])
    marcosAHPOutage = np.array([])
    mooraAHPOutage = np.array([])
    vikorAHPOutage = np.array([])

    for i in range(5):
        totLatencyEntropyTopsis = 0
        totEnergyEntropyTopsis = 0
        totLatencyEntropyMarcos = 0
        totEnergyEntropyMarcos = 0
        totLatencyEntropyMoora = 0
        totEnergyEntropyMoora = 0
        totLatencyEntropyVikor = 0
        totEnergyEntropyVikor = 0

        totLatencyAHPTopsis = 0
        totEnergyAHPTopsis = 0
        totLatencyAHPMarcos = 0
        totEnergyAHPMarcos = 0
        totLatencyAHPMoora = 0
        totEnergyAHPMoora = 0
        totLatencyAHPVikor = 0
        totEnergyAHPVikor = 0


        totTopsisEntropyOutage = 0
        totMarcosEntropyOutage = 0
        totMooraEntropyOutage = 0
        totVikorEntropyOutage =0

        totTopsisAHPOutage = 0
        totMarcosAHPOutage = 0
        totMooraAHPOutage = 0
        totVikorAHPOutage = 0
        
        totalIter = 50
        for iter in range(totalIter):
            tasks = generateTasks(noOfTasks)
            tasksDecisionMatrix = np.array(convertTasksToMatrix(tasks))
            columns = len(tasksDecisionMatrix[0])
            tasksEntropyWeight = entropy_weights(tasksDecisionMatrix)

            entropyTopsisTaskOrdering = getOrderByTopsis(
                tasksDecisionMatrix, tasksEntropyWeight, [1, -1, 1], columns)
            entropyMarcosTaskOrdering = getOrderByMarcos(
                tasksDecisionMatrix, tasksEntropyWeight, [1, -1, 1], columns)
            entropyMooraTaskOrdering = getOrderByMoora(
                tasksDecisionMatrix, tasksEntropyWeight, [1, -1, 1], columns)
            entropyVikorTaskOrdering = getOrderByVikor(
                tasksDecisionMatrix, tasksEntropyWeight, [1, -1, 1], columns)

            [latencyEntropyTopsis, energyEntropyTopsis,outageET] = matchTaskandFogNode(
                entropyTopsisTaskOrdering, fogEntropyTopsisOrdering)
            [latencyEntropyMarcos, energyEntropyMarcos,outageEMa] = matchTaskandFogNode(
                entropyMarcosTaskOrdering, fogEntropyMarcosOrdering)
            [latencyEntropyMoora, energyEntropyMoora,outageEMo] = matchTaskandFogNode(
                entropyMooraTaskOrdering, fogEntropyMooraOrdering)
            [latencyEntropyVikor, energyEntropyVikor,outageEV] = matchTaskandFogNode(
                entropyVikorTaskOrdering, fogEntropyVikorOrdering)

            totLatencyEntropyTopsis += latencyEntropyTopsis
            totEnergyEntropyTopsis += energyEntropyTopsis
            totLatencyEntropyMarcos += latencyEntropyMarcos
            totEnergyEntropyMarcos += energyEntropyMarcos
            totLatencyEntropyMoora += latencyEntropyMoora
            totEnergyEntropyMoora += energyEntropyMoora
            totLatencyEntropyVikor += latencyEntropyVikor
            totEnergyEntropyVikor += energyEntropyVikor

            totTopsisEntropyOutage+=outageET
            totMarcosEntropyOutage+=outageEMa
            totMooraEntropyOutage+=outageEMo
            totVikorEntropyOutage+=outageEV
            # ======================For AHP============================

            AHPTopsisTaskOrdering = getOrderByTopsis(
                tasksDecisionMatrix, tasksAHPWeight, [1, -1, 1], columns)
            AHPMarcosTaskOrdering = getOrderByMarcos(
                tasksDecisionMatrix, tasksAHPWeight, [1, -1, 1], columns)
            AHPMooraTaskOrdering = getOrderByMoora(
                tasksDecisionMatrix, tasksAHPWeight, [1, -1, 1], columns)
            AHPVikorTaskOrdering = getOrderByVikor(
                tasksDecisionMatrix, tasksAHPWeight, [1, -1, 1], columns)

            [latencyAHPTopsis, energyAHPTopsis,outageAT] = matchTaskandFogNode(
                AHPTopsisTaskOrdering, fogAHPTopsisOrdering)
            [latencyAHPMarcos, energyAHPMarcos,outageAMa] = matchTaskandFogNode(
                AHPMarcosTaskOrdering, fogAHPMarcosOrdering)
            [latencyAHPMoora, energyAHPMoora,outageAMo] = matchTaskandFogNode(
                AHPMooraTaskOrdering, fogAHPMooraOrdering)
            [latencyAHPVikor, energyAHPVikor,outageAV] = matchTaskandFogNode(
                AHPVikorTaskOrdering, fogAHPVikorOrdering)

            totLatencyAHPTopsis += latencyAHPTopsis
            totEnergyAHPTopsis += energyAHPTopsis
            totLatencyAHPMarcos += latencyAHPMarcos
            totEnergyAHPMarcos += energyAHPMarcos
            totLatencyAHPMoora += latencyAHPMoora
            totEnergyAHPMoora += energyAHPMoora
            totLatencyAHPVikor += latencyAHPVikor
            totEnergyAHPVikor += energyAHPVikor

            totTopsisAHPOutage+=outageAT
            totMarcosAHPOutage+=outageAMa
            totMooraAHPOutage+=outageAMo
            totVikorAHPOutage+=outageAV


        taskCount = np.append(taskCount, noOfTasks)
        topsisAHPLatency = np.append(
            topsisAHPLatency, totLatencyAHPTopsis/totalIter)
        topsisAHPEnergy = np.append(
            topsisAHPEnergy, totEnergyAHPTopsis/totalIter)

        marcosAHPLatency = np.append(
            marcosAHPLatency, totLatencyAHPMarcos/totalIter)
        marcosAHPEnergy = np.append(
            marcosAHPEnergy, totEnergyAHPMarcos/totalIter)

        mooraAHPLatency = np.append(
            mooraAHPLatency, totLatencyAHPMoora/totalIter)
        mooraAHPEnergy = np.append(mooraAHPEnergy, totEnergyAHPMoora/totalIter)

        vikorAHPLatency = np.append(
            vikorAHPLatency, totLatencyAHPVikor/totalIter)
        vikorAHPEnergy = np.append(vikorAHPEnergy, totEnergyAHPVikor/totalIter)

        topsisEntropyLatency = np.append(
            topsisEntropyLatency, totLatencyEntropyTopsis/totalIter)
        topsisEntropyEnergy = np.append(
            topsisEntropyEnergy, totEnergyEntropyTopsis/totalIter)
        marcosEntropyLatency = np.append(
            marcosEntropyLatency, totLatencyEntropyMarcos/totalIter)
        marcosEntropyEnergy = np.append(
            marcosEntropyEnergy, totEnergyEntropyMarcos/totalIter)
        mooraEntropyLatency = np.append(
            mooraEntropyLatency, totLatencyEntropyMoora/totalIter)
        mooraEntropyEnergy = np.append(
            mooraEntropyEnergy, totEnergyEntropyMoora/totalIter)
        vikorEntropyLatency = np.append(
            vikorEntropyLatency, totLatencyEntropyVikor/totalIter)
        vikorEntropyEnergy = np.append(
            vikorEntropyEnergy, totEnergyEntropyVikor/totalIter)

        topsisEntropyOutage=np.append(topsisEntropyOutage,totTopsisEntropyOutage/totalIter)
        marcosEntropyOutage = np.append(marcosEntropyOutage, totMarcosEntropyOutage/totalIter)
        mooraEntropyOutage = np.append(mooraEntropyOutage, totMooraEntropyOutage/totalIter)
        vikorEntropyOutage = np.append(vikorEntropyOutage, totVikorEntropyOutage/totalIter)

        topsisAHPOutage = np.append(topsisAHPOutage, totTopsisAHPOutage/totalIter)
        marcosAHPOutage = np.append(marcosAHPOutage, totMarcosAHPOutage/totalIter)
        mooraAHPOutage = np.append(mooraAHPOutage, totMooraAHPOutage/totalIter)
        vikorAHPOutage = np.append(vikorAHPOutage, totVikorAHPOutage/totalIter)







        noOfTasks = noOfTasks+200

    generatePlot2(taskCount, topsisEntropyEnergy, topsisAHPEnergy, marcosEntropyEnergy, marcosAHPEnergy, mooraEntropyEnergy, mooraAHPEnergy, vikorEntropyEnergy, vikorAHPEnergy, "Task Count",
                  "System Energy(J)", "with Entropy TOPSIS", "with AHP TOPSIS", "with Entropy MARCOS", "with AHP MARCOS", "with Entropy MOORA", "with AHP MOORA", "With Entropy VIKOR", "With AHP VIKOR")

    generatePlot2(taskCount, topsisEntropyLatency, topsisAHPLatency, marcosEntropyLatency, marcosAHPLatency, mooraEntropyLatency, mooraAHPLatency, vikorEntropyLatency, vikorAHPLatency,
                  "Task Count", "System Latency (in Sec)", "with Entropy TOPSIS", "with AHP TOPSIS", "with Entropy MARCOS", "with AHP MARCOS", "with Entropy MOORA", "with AHP MOORA", "With Entropy VIKOR", "With AHP VIKOR")
    generatePlot2(taskCount, topsisEntropyOutage, topsisAHPOutage, marcosEntropyOutage, marcosAHPOutage, mooraEntropyOutage, mooraAHPOutage, vikorEntropyOutage, vikorAHPOutage, "Task Count",
                  "Outage Count", "with Entropy TOPSIS", "with AHP TOPSIS", "with Entropy MARCOS", "with AHP MARCOS", "with Entropy MOORA", "with AHP MOORA", "With Entropy VIKOR", "With AHP VIKOR")


if __name__ == "__main__":
    main()
