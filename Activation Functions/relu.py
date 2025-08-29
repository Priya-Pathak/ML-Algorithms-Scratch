# Resources used
# 1. https://www.youtube.com/watch?v=KUrBgoIEQK0&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=6
# 2. https://www.youtube.com/watch?v=90Kxtd8zCHI&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=7

# Notes:
# Does it qualify the list?
# 1. Non-linearity : Yes
# 2. Differentiable : Yes
# 3. Range : Yes
# 4. Monotonic : Yes
# 5. Computationaly efficient : Yes


# Relu Activation Function
# Formula:
# f(x) = max(0,x)
# Mapping : (-inf, +inf) -->[0, +inf]
# HyperParameter : a,b for changing the slope and intercept of the relu
# Use Cases: For hidden layers
# Differentiated : 1 when x>0, 0 otherwise, for x=0 it is not defined
# Drawback: [Exploding gradient problem, Dead neurons(sparsity acts as regularizer)]

import numpy as np

class Relu_function():
    
    def __init__(self):
        pass
    
    def activate(self, x):
        print('x value: ',x)
        x = max(0,x)
        print('Relu value: ',x)

relu_function_1 = Relu_function()
relu_function_1.activate(x=32)
relu_function_1.activate(x=-8)

# Variants of relu:
# 1. Leaky relu: f(x) = max(0.1x,x) [0.1 value is a hyperparameter like learning rate]
# 2. Randomized leaky relu : The hyperparameter above is taken randomly from a uniform distribution (0,1)
#    During the test time we take the average value of upper_limit and lower_limit and use that.
# 3. Parameteric relu: f(x) = max(ax, x) the model learns the a. It is now a parameter and not a hyper-parameter
# 4. Exponential Linear Unit: ELU = e(x)-1 when x<0 otherwise x
# 5. Scaled Exponential Linear Unit: SELU = p[a*e(x)-a when x<0 otherwise x] : Here p and a are fixed