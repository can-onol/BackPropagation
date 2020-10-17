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

temp_initial_inputs = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
temp_expected_output = np.array([0, 1, 1, 0])
'''for i in range(4):
    print("i = ", i)
    print(temp_initial_inputs[0,i], temp_initial_inputs[1,i], ":", temp_expected_output[i])
'''

# Initial weights for hidden layer
w11 = random()
w12 = random()
w21 = random()
w22 = random()

# Initial weights for output layer
w31 = random()
w32 = random()

# Initial weights for biases
w1b = random()
w2b = random()
w3b = random()
# Initial biases
b1 = 1
b2 = 1

# Gaussian Noise
mu, sigma = 0, 0.1

epochs = 100000
learning_rate = 0.02
graph_error = []
dim = []
for i in range(epochs):
    #print("Epochs is", i)
    # generate an integer random variable between 0 and 3 in order to select input pair
    index = randint(0, 3)
    noise1 = np.random.normal(mu, sigma)
    noise2 = np.random.normal(mu, sigma)
    i1 = temp_initial_inputs[0,index] # select an input pair out of 4 pairs
    i2 = temp_initial_inputs[1, index]
    expected_output = temp_expected_output[index] # select expected output corresponding to input
    #print(i1, i2, ":", expected_output)
    i1= i1 + noise1 # adding gaussian noise
    i2 = i2 + noise2 # adding gaussian noise

    # Forward Propagation
    h1 = w11*i1 + w12*i2 + w1b*b1
    h2 = w21*i1 + w22*i2 + w2b*b1
    # Sigmoid act. func. was used for hidden layer
    h1_output = sigmoid(h1)
    h2_output = sigmoid(h2)
    y = w31*h1_output + w32*h2_output + w3b*b2
    #Linear act. func. was used instead of sigmoid for output layer
    y_output = linear_function(y)

    # Back Propagation
    error = expected_output - y_output
    graph_error.append(error)
    dim.append(i)
    #print("Error", error)
    # Derivative of Linear act. func. was used
    delta_error = error * derivative_linear(y_output) # Partial derivative dE/dxj
    # Derivative of Sigmoid act. func. was used
    delta_h1 = delta_error * w31 * derivative_sigmoid(h1_output) # Partial derivative dE/dxj * dxj/dxi
    delta_h2 = delta_error * w32 * derivative_sigmoid(h2_output)

    # Updating weights for hidden layer
    w31 = w31 + delta_error * h1_output * learning_rate
    w32 = w32 + delta_error * h2_output * learning_rate
    w3b = w3b + delta_error * b2 * learning_rate

    # Updating weights for input layer
    w11 = w11 + delta_h1 * i1 * learning_rate
    w12 = w12 + delta_h1 * i2 * learning_rate
    w21 = w21 + delta_h2 * i1 * learning_rate
    w22 = w22 + delta_h2 * i2 * learning_rate
    w1b = w1b + delta_h1 * b1 * learning_rate
    w2b = w2b + delta_h2 * b2 * learning_rate

#print("Predicted output = ", y_output)

for i in range (20):
    print(i + 1)
    for j in range(4):
        noise1 = np.random.normal(mu, sigma)
        noise2 = np.random.normal(mu, sigma)
        i1 = temp_initial_inputs[0, j]  # select an input pair out of 4 pairs
        i2 = temp_initial_inputs[1, j]
        expected_output = temp_expected_output[j]
        i1_noise = i1 + noise1  # adding gaussian noise
        i2_noise = i2 + noise2  # adding gaussian noise

        # Forward Propagation
        h1 = w11 * i1 + w12 * i2 + w1b * b1
        h2 = w21 * i1 + w22 * i2 + w2b * b1
        # Sigmoid act. func. was used for hidden layer
        h1_output = sigmoid(h1)
        h2_output = sigmoid(h2)
        y = w31 * h1_output + w32 * h2_output + w3b * b2
        # Linear act. func. was used instead of sigmoid for output layer
        y_output = linear_function(y)
        error = expected_output - y_output
        print("Index: ", j, "Inputs: ", i1, i2, "Inputs with Noise: ", np.around(i1_noise, 3), np.around(i2_noise, 3), "Output: ",
              expected_output, "Predicted Output: ", np.around(y_output, 3), "Error: ",
              np.around(error, 3))

'''
hidden_weights = np.array([[w11, w12],[w21, w22]])
hidden_bias_weights = np.array([[w1b],[w2b]])
output_weights = np.array([[w31, w32]])
output_bias_weights = w3b
#predicted_output = np.zeros((1,4))
#error_forward = np.zeros((4))

for i in range (3):
    for j in range(4):
        noise = np.random.normal(mu, sigma, [2, 1])
        inputs = temp_initial_inputs[:, j]  # select an input pair out of 4 pairs
        inputs_noise = inputs + noise
        hidden_layer = np.dot(hidden_weights, inputs_noise) + np.dot(hidden_bias_weights, b1)
        hidden_layer_output = sigmoid(hidden_layer)

        output_layer = np.dot(output_weights, hidden_layer_output) + output_bias_weights*b2
        predicted_output = linear_function(output_layer)
        expected_output = temp_expected_output[j]

        error_forward = expected_output - predicted_output
        print("Index: ", j, "Input: ", inputs, "Inputs with Noise: ", np.around(inputs_noise,3), "Output: ", expected_output, "Predicted Output: ", np.around(predicted_output,3), "Error: ", np.around(error_forward,3))
'''
'''
plt.plot(dim, graph_error)
p.show()
'''
