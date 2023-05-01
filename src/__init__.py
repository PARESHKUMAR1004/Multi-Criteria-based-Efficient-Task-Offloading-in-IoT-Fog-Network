import numpy as np
# from normalisations import minmax_normalization
# print("Hiii")
# finalTaskTable = np.zeros(10, np.ndarray)
# finalTaskTable[2] = np.array([2, 4, 5])
# print(finalTaskTable)
# print(finalTaskTable[2])
# if isinstance(2,int):
#     print("Hii")
# print(type([1,2,4]))

Dict = {}
print(type(Dict))
if Dict.get(33.5)==None:
    Dict[33.5]=[]
    Dict[33.5].append([1,2,3])
else:
    Dict[33.5].append([5,6,7])
Dict[33.5].append([8,2,4])
print(Dict)
