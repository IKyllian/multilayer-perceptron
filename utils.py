import pandas as pandas
from sklearn.model_selection import train_test_split
import os
import statistics
import numpy as np 

TRAIN_DATAFRAME_FILE = 'trainDataFrame.csv'
PREDICTION_DATAFRAME_FILE = 'predictionDataFrame.csv'
FOLDER = './datasets'
# COLUMNS = [
#     'id',
#     'diagnosis'
#     "radius"
# 	"texture"
# 	"perimeter"
# 	"area"
# 	"smoothness"
# 	"compactness"
# 	"concavity"
# 	"concave points"
# 	"symmetry" 
# 	"fractal dimension"
# ]
COLUMNS = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

def openCsv(path: str) -> pandas.DataFrame:
    fileDataFrame = pandas.read_csv(path)
    return fileDataFrame

def predictionSubSet(dataFrame: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    length = dataFrame.shape[0]
    trainDataFrame, predictionSet = train_test_split(dataFrame, test_size=int(length/2), train_size=int(length/2), shuffle=False)
    # dataFrameSplit = [dataFrame.iloc[:-length/2], dataFrame.iloc[-length/2:]]
    # predictionSet = dataFrameSplit[1]
    # trainDataFrame = dataFrameSplit[0]
    trainDataFrame.columns = COLUMNS
    predictionSet.columns = COLUMNS
    return (trainDataFrame, predictionSet)

def exitWithMessage(message: str):
    print(message)
    exit()

def checkFileValidity(filePath: str):
    split_tup = os.path.splitext(filePath)
    file_extension = split_tup[len(split_tup) - 1]
    if (file_extension != '.csv'):
        exitWithMessage("Bad extension file")
    if not os.path.isfile(filePath):
        exitWithMessage("File does not exist")

def parseFileArg(argv, argc) -> str:
    if (argc < 2):
        exitWithMessage("A dataset file is needed")
    filePath = argv[1]
    checkFileValidity(filePath)
    return filePath

def standardizeDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    newDataFrame = {}
    for col in dataFrame:
        meanValue = statistics.mean(dataFrame[col])
        stdValue = np.std(dataFrame[col])
        standardiweCol = (dataFrame[col] - meanValue) / stdValue
        newDataFrame.update({col: standardiweCol})
    return pandas.DataFrame(newDataFrame)