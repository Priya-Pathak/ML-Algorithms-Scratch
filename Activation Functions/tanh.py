# Resources used
# 1. https://www.youtube.com/watch?v=HeJ_D7CxruM&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=4

# Notes:
# Does it qualify the list?
# 1. Non-linearity : Yes
# 2. Differentiable : Yes
# 3. Range : Yes
# 4. Monotonic : Yes
# 5. Computationaly efficient : Takes longer training time


# Tanh Activation Function
# Formula:
# f(x) = [e^(x)-e^(-x)]/[e^(x)+e^(-x)]
# Mapping : (-inf, +inf) --> [-1, 1]
# HyperParameter : theta(could be used) if probability < theta then 0 else classify 1
# Use Cases: For hidden layers activation
# Differentiated : Bell curve
# Drawback: [saturated at ends -inf and +inf, Vanishing Gradient]

import numpy as np

class Tanh_function():
    
    def __init__(self, theta= 0):
        self.theta = theta
    
    def activate(self, x):
        x = (np.exp(x)-np.exp(-x))/ (np.exp(x)+np.exp(-x))
        print('Tanh value: ',x)
        
        if x < self.theta:
            print('Result with theta=',self.theta, ' : ',0)
        else:
            print('Result with theta=',self.theta, ' : ',1)

tanh_function_1 = Tanh_function(theta=0.5)
tanh_function_1.activate(x=32)
tanh_function_1.activate(x=-8)

tanh_function_2 = Tanh_function(theta=0.7)
tanh_function_2.activate(x=32)
tanh_function_2.activate(x=-8)