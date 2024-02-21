import numpy as np

'''
MIT License

Copyright (c) 2024 HarGrif

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''


# Container class for whole neural net
class Neural_Net:
    def __init__(self):
        self.layers = None
        self.neurons = None
        self.out = None

    # Load a previous network from known bias and weights
    def load_net(self, net):
        self.layers = [NN_Layer() for i in range(len(net))]
        for layer, data in zip(self.layers, net):
            layer.load_layer(data)

    # Initialise a fresh network when no previous bias or weights known
    def fresh_start(self, n_inputs, n_neurons, n_outputs):
        # Adds parameters for final layer
        n_neurons.append(n_outputs)
        # Initialises layers
        self.layers = [NN_Layer() for i in range(len(n_neurons))]
        self.neurons = n_neurons
        i = 0
        n_inputs = list([n_inputs]) + self.neurons[0:-1]
        # Initialise each layer with given number of neurons
        for layer, n_neurons, inputs in zip(self.layers, self.neurons, n_inputs):
            layer.fresh_layer(inputs, n_neurons)

    # Actually runs the net
    def run(self, layer_input):
        i = 0
        final_layer = False
        for layer in self.layers:
            # Checks if it is the final layer
            if i == len(self.layers)-1:
                final_layer = True
            layer.forward(layer_input, final_layer)
            # Sets next layer input to previous layer output
            layer_input = layer.output
            i += 1
        # Outputs final layer output
        self.out = layer_input

    # Save your network
    def save_net(self):
        net_data = []
        for layer in self.layers:
            data = np.vstack([layer.biases, layer.weights])
            net_data.append(data.tolist())
        return net_data


# Subclass for each layer of the neural net
class NN_Layer:
    def __init__(self):
        self.output = None
        self.weights = None
        self.biases = None

    # Load a layer
    def load_layer(self, data):
        # Initialise weights from data
        self.weights = data[1:]
        # Initialise biases from data
        self.biases = data[0]

    # Initialise a fresh layer
    def fresh_layer(self, inputs, n_neurons):
        # Initialise weights as gaussian dist normalised around 1 scaled down by a factor of 0.1
        self.weights = 0.1 * np.random.randn(int(inputs), int(n_neurons))
        # Initialise all biases as zero
        self.biases = np.zeros((1, n_neurons))

    # Forward pass of nn
    def forward(self, inputs, final_layer):
        # Take sum of the products of weights and inputs + biases
        self.output = np.dot(inputs, self.weights) + self.biases
        # Doesn't ReLU the final layer
        if not final_layer:
            # ReLU
            self.output = np.maximum(0, self.output)

