from numpy import random


class Task:
    def __init__(self, taskSize, taskDeadline, outputSize):
        self.taskSize = taskSize
        self.taskDeadline = taskDeadline
        self.outputSize = outputSize

    def showTaskAttributes(self):
        print("Task Size: ", self.taskSize)
        print("Task Deadline: ", self.taskDeadline)
        print("Task outputSize: ", self.outputSize)


def generateTasks(count):
    tasks = []
    for i in range(count):
        size = random.randint(2000, 5000)
        deadline = random.randint(3000, 6000)
        outputSize = size/(random.randint(4, 8))
        task = Task(size, deadline, outputSize)
        tasks.append(task)
    return tasks


def convertTasksToMatrix(tasks):
    DecisionMatrix = []
    for i in range(len(tasks)):
        task = []
        task.append(tasks[i].taskSize)
        task.append(tasks[i].taskDeadline)
        task.append(tasks[i].outputSize)
        DecisionMatrix.append(task)

    return DecisionMatrix
