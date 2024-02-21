import numpy as np
from class_neural_net import Neural_Net

# Give it a previous network as a lolol: network -> layers -> bias + weights (bias is top row of matrix)
# This network takes 3 pieces of input data
# It has two hidden layers with 4 and 3 neurons respectively and an output layer with 2 neurons
my_net = [[[0.1, 0.5, -0.3, 0.4],  # This is hidden layer 1 bias
          [0.05, -0.3, 0.2, 1.3],  # This is hidden layer 1 weights
          [0.1, 0.03, -0.2, -0.1],  # This is also hidden layer 1 weights
          [-0.2, -0.05, 0.1, 0.1]],  # This is also hidden layer 1 weights
          [[0, 0.2, -0.1],  # This is hidden layer 2 bias
          [0.05, 0.2, 0.5],  # This is hidden layer 2 weights
          [-0.1, -0.04, 0.3],  # This is also hidden layer 2 weights
          [0.1, 0.03, -0.2],  # This is also hidden layer 2 weights
          [0.3, 0.1, -0.02]],  # This is also hidden layer 2 weights
          [[0.3, 0.4],  # This is output layer bias
          [0.2, -0.2],  # This is output layer weights
          [0.1, 0.2],  # This is also output layer weights
          [0.3, -0.1]]]  # This is also output layer weights

# Generate random input data
X = np.random.randn(3)

# Build class
nn = Neural_Net()
# Initialise layer weights and biases
nn.load_net(my_net)
# # Run net with random data X
nn.run(X)

print(nn.out)
