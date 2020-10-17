# Backpropagation Algorithm for Learning XOR Gate
# Sigmoid Function for Hidden Layer
# Linear Function for Output Layer
# 2 input, 2 hidden, 1 output neurons
# Hidden and Output Wights Random
# Hidden and Output Biases 1
# 10000 Iteration and 0.1 Learning Rate
# Gaussian Noise for Input mean is 0 standart deviation 0.1

from random import seed
from random import randint
from random import random
import matplotlib.pyplot as plt
import pylab as p
import numpy as np

# seed random number generator
seed(1)

def linear_function(x):
    return x

def derivative_linear(x):
    return 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)

inputs = []
outputs = []
startPoint, endPoint = 0.1, 0.6

for a in np.arange(startPoint, endPoint, 0.1):
    for b in np.arange(startPoint, endPoint, 0.1):
        inputs.append([a, b])
        z = (1 - a) ** 2 + 100 * (b - a ** 2) ** 2
        outputs.append([z])

temp_initial_inputs = np.array(inputs) # (endPoint*10 - startPoint*10)^2 x 2 matrix
temp_expected_output = np.array(outputs) # (endPoint*10 - startPoint*10)^2 x 1 matrix eg. 25x1

# Initial biases
b1 = 1
b2 = 1

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1


# Gaussian Noise
mu, sigma = 0, 0

epochs = 100000
learning_rate = 0.000001

hidden_weights = np.random.uniform(size=(hiddenLayerNeurons, inputLayerNeurons))
hidden_bias_weights = np.random.uniform(size=(hiddenLayerNeurons, 1))
output_weights = np.random.uniform(size=(outputLayerNeurons, hiddenLayerNeurons))
output_bias_weights = np.random.uniform(size=(outputLayerNeurons, 1))
index = 0
item = 10
lastIndex = ((endPoint-startPoint)*10)**2 - 1
for i in range(epochs):
    if index+item == lastIndex:
        index = 0
    if index+item > lastIndex:
        index = lastIndex - item
    #print("Epochs is", i)
    # Generate an integer random variable between 0 and (rows of inputs) - 1 in order to select input pair
    #index = randint(0, ((endPoint-startPoint)*10)**2 - (item+1)) # Index can exceed inputs array if index near to last 10 terms
    noise = np.random.normal(mu, sigma, [item, 2]) # Generate Gaussian Noise for adding inputs
    initial_inputs = temp_initial_inputs[index:index+item, :] # select an input pair out of 4 pairs 1x2 matrix
    expected_output = temp_expected_output[index:index+item] # select expected output corresponding to input

    inputs = initial_inputs # 1x2 matrix

    # Forward Propagation
    hidden_layer = np.dot(hidden_weights, inputs.T) + np.dot(hidden_bias_weights, b1)
    # Sigmoid act. func. was used for hidden layer
    hidden_layer_activation = sigmoid(hidden_layer)  # #hiddenLayerNeurons x 1 matrix

    output_layer = np.dot(output_weights, hidden_layer_activation) + np.dot(output_bias_weights, b2)
    # Linear act. func. was used instead of sigmoid for output layer
    output_layer_activation = linear_function(output_layer) # #outputLayerNeurons x 1 matrix

    # Back Propagation
    # Mean Squared Error
    error_train = np.square(np.subtract(temp_expected_output[index:index+10], output_layer_activation)).mean()

    # Derivative of Linear act. func. was used
    delta_error = error_train * derivative_linear(output_layer_activation)  # Partial derivative dE/dxj 1x1 matrix

    # Derivative of Sigmoid act. func. was used
    delta_hidden_layer = output_weights.T * derivative_sigmoid(hidden_layer_activation) * delta_error  # Partial derivative dE/dxj * dxj/dxi #hiddenLayerNeuronsx1 matrix

    # Updating weights for hidden layer
    output_weights = output_weights + hidden_layer_activation.T * delta_error * learning_rate # #1xhiddenLayerNeurons matrix
    output_bias_weights = output_bias_weights + delta_error * b2 * learning_rate

    # Updating weights for input layer
    hidden_weights = hidden_weights + np.dot(delta_hidden_layer, inputs) * learning_rate
    hidden_bias_weights = hidden_bias_weights + delta_hidden_layer * b1 * learning_rate
    print("Error", error_train)
    index += 5


