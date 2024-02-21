# Neural Network From Scratch
This Python code provides a simple implementation of a neural network using just the NumPy library.

It will genererate a random new neural network with random weight and no bias or it will load a previous Nueral Network. 

This is another project as part of my Final Year Project on GAs at University.

## Features
- Only has numpy as a prerequisite.
- Flexible architecture for defining the number of inputs, neurons, and outputs.
- Option to load a pre-trained network from known biases and weights or start fresh.
- Initialization of layers with Gaussian-distributed weights and zero biases.
- Utilizes the Rectified Linear Unit (ReLU) activation function for hidden layers.
- Will run the network from given data.
- ** Does not train the network **

## Usage
The neural network consists of a container class `Neural_Net` and a subclass `NN_Layer` for individual layers.

There are two example files of how to use it. [This one](example_generate_fresh_network.py) generates a fresh network from a given number of neurons, input data size and output size. 
[This one](example_load_previous_network.py) loads a previously trained network.

