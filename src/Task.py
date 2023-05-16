from numpy import random


class Task:

    # taskType=[1,-1,1]
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
    i = 0
    while i < count:
        size = random.randint(300, 600)*8*1024  # [300,600]KB
        # [15,25] sec , Got from Chitta Sir's Paper
        deadline = random.uniform(0.7, 1.2)
        outputSize = size/(random.randint(4, 8))
        task = Task(size, deadline, outputSize)
        if isinstance(task, int):
            continue
        i += 1
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
