import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
import sys

def main():
    filePath = parseFileArg(sys.argv, len(sys.argv))
    dataframe = pandas.read_csv(filePath)
    trainDataFrame, predictionSet = predictionSubSet(dataframe)
    trainDataFrame.to_csv(FOLDER + '/' + TRAIN_DATAFRAME_FILE, index=False)
    predictionSet.to_csv(FOLDER + '/' + PREDICTION_DATAFRAME_FILE, index=False)

if (__name__ == "__main__"):
    main()