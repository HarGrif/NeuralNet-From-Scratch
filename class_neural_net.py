import numpy as np


# Container class for whole neural net
class Neural_Net:
    def __init__(self, n_neurons, n_outputs):
        # Adds parameters for final layer
        n_neurons.append(n_outputs)
        # Initialises layers
        self.layers = [NN_Layer() for i in range(len(n_neurons))]
        self.neurons = n_neurons
        self.out = None

    # Initialise each layer when no previous bias, or weights known
    def fresh_start(self, n_inputs):
        i = 0
        n_inputs = list([n_inputs]) + self.neurons[0:-1]
        # Initialise each layer with given number of neurons
        for layer, n_neurons, inputs in zip(self.layers, self.neurons, n_inputs):
            layer.init_layer_scratch(inputs, n_neurons)

    # Actually runs the net
    def run(self, layer_input):
        i = 0
        final_layer = False
        for layer in self.layers:
            if i == len(self.layers)-1:
                final_layer = True
            layer.forward(layer_input, final_layer)
            layer_input = layer.output
            i += 1
        self.out = layer_input


# Subclass for each layer of the neural net
class NN_Layer:
    def __init__(self):
        self.output = None
        self.weights = None
        self.biases = None

    # Initialise a fresh layer
    def init_layer_scratch(self, inputs, n_neurons):
        # Initialise weights as gaussian dist normalised around 1 scaled down by a factor of 0.1
        self.weights = 0.1 * np.random.randn(int(inputs), int(n_neurons))
        # Initialise all biases as zero
        self.biases = np.zeros((1, n_neurons))

    # Forward pass of nn
    def forward(self, inputs, final_layer):
        # Take sum of the products of weights and inputs + biases
        self.output = np.dot(inputs, self.weights) + self.biases
        if not final_layer:
            # ReLU
            self.output = np.maximum(0, self.output)

