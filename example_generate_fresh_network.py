import numpy as np
from class_neural_net import Neural_Net

# Assign network parameters
n_neurons = [15, 10]  # Number of neurons per layer, size of array dictates number of layers
n_inputs = 10  # Size of input data
n_outputs = 3  # Number of outputs you want

# Generate random input data
X = np.random.randn(n_inputs)

# Build class
nn = Neural_Net()
# Build framework and initialise layer weights and biases
nn.fresh_start(n_inputs, n_neurons, n_outputs)
# Run net with random data X
nn.run(X)

print(nn.out)

