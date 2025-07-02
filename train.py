import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from mlp import *

def main():
    filePath = FOLDER + '/' + TRAIN_DATAFRAME_FILE
    checkFileValidity(filePath)
    dataframe = pandas.read_csv(filePath)
    diagnosis = dataframe['diagnosis']
    dataframe.drop(columns=["id", "diagnosis"], inplace=True)
    x = standardizeDataFrame(dataFrame=dataframe)
    y = diagnosis.map({'M': 1, 'B': 0}).values
    mlp = MLP(inputs=x, labels=y, hidden_layer_size=2, input_layer_size=x.shape[1])
    mlp.train()

if (__name__ == "__main__"):
    main()