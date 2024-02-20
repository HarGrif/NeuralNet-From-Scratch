import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class NN_Layer:
    def __init__(self, n_inputs, n_neurons):
        # Initialise weights as gaussian dist normalised around 1 scaled down by a factor of 0.1
        self.output = None
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialise all biases as zero
        self.biases = np.zeros((1, n_neurons))

    # Forward pass of nn
    def forward(self, inputs):
        # Take sum of the products of weights and inputs + biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = NN_Layer(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)