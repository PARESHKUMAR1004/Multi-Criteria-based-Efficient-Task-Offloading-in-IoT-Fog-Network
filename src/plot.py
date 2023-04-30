import matplotlib.pyplot as plt


def generatePlot(X, Y1, Y2, xlabel, ylabel, legend1, legend2):
    print("X: ", X)
    print("Y1: ", Y1)
    print("Y2: ", Y2)
    plt.plot(X, Y1)
    plt.plot(X, Y2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend1, legend2])

    plt.show()
