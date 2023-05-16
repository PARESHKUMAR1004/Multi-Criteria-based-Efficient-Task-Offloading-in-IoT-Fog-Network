import matplotlib.pyplot as plt
import numpy as np


def generatePlot(X, Y1, Y2,Y3,Y4, xlabel, ylabel, legend1, legend2,legend3,legend4):
    
    barWidth = 0.2
    fig = plt.subplots(figsize=(12, 8))

    br1 = np.arange(len(X))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]


    plt.bar(br1, Y1, color='r', width=barWidth,
            edgecolor='grey', label='Y1')


    plt.bar(br2, Y2, color='g', width=barWidth,
        edgecolor='grey', label='Y2')
    
    plt.bar(br3, Y3, color='b', width=barWidth,
            edgecolor='grey', label='Y3')

    plt.bar(br4, Y4, color='y', width=barWidth,
        edgecolor='grey', label='Y4')


    # plt.bar(X, Y1,width=0.3)
    # plt.bar(X, Y2,width=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend1, legend2,legend3,legend4])
    # plt.ylim(min(np.min(Y1), np.min(Y2))-1e7, max(np.max(Y1), np.max(Y2))+1e7)
    plt.xticks([r + barWidth for r in range(len(X))],X)

    plt.show()


def generatePlot2(X, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, xlabel, ylabel, legend1, legend2, legend3, legend4, legend5, legend6, legend7, legend8):

    barWidth = 0.1
    fig = plt.subplots(figsize=(12, 8))

    br1 = np.arange(len(X))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x + barWidth for x in br6]
    br8 = [x + barWidth for x in br7]

    plt.bar(br1, Y1, color='r',hatch='/', width=barWidth,
            edgecolor='grey', label='Y1')

    plt.bar(br2, Y2, color='r', hatch='.0', width=barWidth,
            edgecolor='grey', label='Y2')

    plt.bar(br3, Y3, color='b', hatch='/', width=barWidth,
            edgecolor='grey', label='Y3')

    plt.bar(br4, Y4, color='b', hatch='.0', width=barWidth,
            edgecolor='grey', label='Y4')
    plt.bar(br5, Y5, color='y', hatch='/', width=barWidth,
            edgecolor='grey', label='Y5')
    
    plt.bar(br6, Y6, color='y', hatch='.0', width=barWidth,
            edgecolor='grey', label='Y6')
    
    plt.bar(br7, Y7, color='g', hatch='/', width=barWidth,
            edgecolor='grey', label='Y7')
    
    plt.bar(br8, Y8, color='g', hatch='.0', width=barWidth,
            edgecolor='grey', label='Y8')
    
    

    # plt.bar(X, Y1,width=0.3)
    # plt.bar(X, Y2,width=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend1, legend2, legend3, legend4,
               legend5, legend6, legend7, legend8])
    # plt.ylim(min(np.min(Y1), np.min(Y2))-1e7, max(np.max(Y1), np.max(Y2))+1e7)
    plt.xticks([r + barWidth for r in range(len(X))], X)

    plt.show()
#     plt.savefig(ylabel)
#     plt.close()
