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
import time

x = np.linspace(-2, 1.9, 40)

# numpy.linspace creates an array of
# 9 linearly placed elements between
# -4 and 4, both inclusive
y = np.linspace(-2, 1.9, 40)

# The meshgrid function returns
# two 2-dimensional arrays
x_1, y_1 = np.meshgrid(x, y)
random_data = np.random.random((40, 40))
plt.contourf(x_1, y_1, random_data, cmap='jet')

plt.colorbar()
plt.show()
