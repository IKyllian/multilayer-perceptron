import pandas as pandas
import matplotlib.pyplot as plt
from utils import *

def main():
    filePath = FOLDER + '/' + TRAIN_DATAFRAME_FILE
    checkFileValidity(filePath)
    dataframe = pandas.read_csv(filePath)
    dataframe = standardizeDataFrame(dataFrame=dataframe)
    # print(dataframe[0])

if (__name__ == "__main__"):
    main()