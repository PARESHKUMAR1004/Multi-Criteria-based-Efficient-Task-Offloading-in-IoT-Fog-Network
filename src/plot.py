import matplotlib.pyplot as plt
import numpy as np


def generatePlot(X, Y1, Y2, xlabel, ylabel, legend1, legend2):
    

    # print("X: ", X)
    # print("Y1: ", Y1)
    # print("Y2: ", Y2)
    # print("==============================")

    plt.plot(X, Y1)
    plt.plot(X, Y2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend1, legend2])

    plt.show()
#     barWidth = 10

#     fig = plt.subplots(figsize=(12, 8))

# # set height of bar


# # Set position of bar on X axis
#     br1 = np.arange(len(Y1))
#     br2 = [x + barWidth for x in br1]


# # Make the plot
#     plt.bar(br1, Y1, color='r', width=barWidth,
#             edgecolor='grey', label='IT')
#     plt.bar(br2, Y2, color='g', width=barWidth,
#             edgecolor='grey', label='ECE')

# # # Adding Xticks
# #     plt.xlabel('Task', fontweight='bold', fontsize=15)
# #     plt.ylabel('Students passed', fontweight='bold', fontsize=15)
# #     plt.xticks([r + barWidth for r in range(len(IT))],
# #            ['2015', '2016', '2017', '2018', '2019'])

#     plt.legend()
#     plt.show()
