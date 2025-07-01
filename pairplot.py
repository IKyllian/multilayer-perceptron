import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns

def main():
    filePath = FOLDER + '/' + TRAIN_DATAFRAME_FILE
    checkFileValidity(filePath)
    dataframe = pandas.read_csv(filePath)
    subset = ['diagnosis', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']

    sns.pairplot(dataframe[subset], hue='diagnosis', diag_kind='hist', markers=['o', 's'])
    plt.show()

if (__name__ == "__main__"):
    main()