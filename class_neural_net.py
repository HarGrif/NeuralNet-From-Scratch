import numpy as np


class Neural_Net:
    def __init__(self, n_layers, n_neurons):
        self.layers = [NN_Layer() for i in range(n_layers + 1)]
        self.neurons = np.concatenate(n_neurons, 8)

    def fresh_start(self, n_inputs):
        # Initialise each layer
        for layer in self.layers:
            layer.init_layer_scratch(n_inputs, self.neurons)

    def run(self, X):
        for layer in self.layers:
            layer.forward(X)


class NN_Layer:
    def __init__(self):
        self.output = None
        self.weights = None
        self.biases = None

    # Initialise a fresh layer
    def init_layer_scratch(self, n_inputs, n_neurons):
        # Initialise weights as gaussian dist normalised around 1 scaled down by a factor of 0.1
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialise all biases as zero
        self.biases = np.zeros((1, n_neurons))

    # Forward pass of nn
    def forward(self, inputs):
        # Take sum of the products of weights and inputs + biases
        self.output = np.dot(inputs, self.weights) + self.biases
        # ReLU
        self.output = np.maximum(0, self.output)

