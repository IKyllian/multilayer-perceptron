import pandas as pandas
from sklearn.model_selection import train_test_split
import os

TRAIN_DATAFRAME_FILE = 'trainDataFrame.csv'
PREDICTION_DATAFRAME_FILE = 'predictionDataFrame.csv'
FOLDER = './datasets'

def openCsv(path: str) -> pandas.DataFrame:
    fileDataFrame = pandas.read_csv(path)
    return fileDataFrame

def predictionSubSet(dataFrame: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    length = dataFrame.shape[0]
    trainDataFrame, predictionSet = train_test_split(dataFrame, test_size=int(length/2), train_size=int(length/2), shuffle=False) #, random_state=41)
    # dataFrameSplit = [dataFrame.iloc[:-length/2], dataFrame.iloc[-length/2:]]
    # predictionSet = dataFrameSplit[1]
    # trainDataFrame = dataFrameSplit[0]
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
