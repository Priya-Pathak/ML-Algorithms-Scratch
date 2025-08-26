# Resources used
# 1. https://www.youtube.com/watch?v=_H5ST6o4doU&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=3

# Notes:
# Does it qualify the list?
# 1. Non-linearity : Yes
# 2. Differentiable : Yes
# 3. Range : Yes
# 4. Monotonic : Yes
# 5. Computationaly efficient : Takes longer training time


# Step Activation Function
# Formula:
# f(x) = 1/(1+e^(-x))
# Mapping : (-inf, +inf) --> [0, 1] range like probability
# HyperParameter : theta(could be used) if probability < theta then 0 else classify 1
# Use Cases: Binary classification when you need probability like results
# Differentiated : Bell curve
# Drawback: [saturated at ends -inf and +inf, Vanishing Gradient, Not-zero centred]

import numpy as np

class Sigmoid_function():
    
    def __init__(self, theta= 0):
        self.theta = theta
    
    def activate(self, x):
        x = 1 / (1+np.exp(x))
        print('Sigmoid value: ',x)
        if x < self.theta:
            print('Result with theta=',self.theta, ' : ',0)
            return 0
        else:
            print('Result with theta=',self.theta, ' : ',1)
            return 1

sigmoid_function_1 = Sigmoid_function(theta=0.5)
sigmoid_function_1.activate(x=32)
sigmoid_function_1.activate(x=-8)

sigmoid_function_2 = Sigmoid_function(theta=0.7)
sigmoid_function_2.activate(x=32)
sigmoid_function_2.activate(x=-8)