import numpy as np
from class_neural_net import Neural_Net, NN_Layer

n_neurons = [15, 10]
n_inputs = 10
n_outputs = 3

X = np.random.randn(n_inputs)

nn = Neural_Net(n_neurons, n_outputs)
nn.fresh_start(n_inputs)
nn.run(X)

print(nn.out)

