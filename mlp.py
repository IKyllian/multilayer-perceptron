import numpy as np
import pandas as pandas

def sigmoid(x) -> float :
    return 1 / (1 + np.exp(-x))
class MLP:
    def __init__(self, inputs: pandas.DataFrame, labels: list[bool], hidden_layer_size, input_layer_size, output_layer_size = 2, learning_rate = 0.0001, batch_size = 8, epochs = 80, ):
        self.inputs = inputs
        self.labels = labels
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.input_layer_size = input_layer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_size = output_layer_size
        
        self.W1 = np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(2 / input_layer_size)
        # print(self.W1)
        self.b1 = np.zeros((1, hidden_layer_size))
        self.W2 = np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(2 / hidden_layer_size)
        # print(self.W2)
        self.b2 = np.zeros((1, output_layer_size))
    
    def forwardPropagation(self):
        print('forwardPropagation')
        self.z1 = np.dot(self.inputs, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        print(self.z1)
        print(self.a1)
        print(self.z2)
        print(self.z2)
        
    def backPropagation():
        print('backPropagation')
    
    def lossFunction():
        print('lossFunction')
    
    def train(self):
        print('train')
        for epoch in self.epochs:
            self.forwardPropagation()