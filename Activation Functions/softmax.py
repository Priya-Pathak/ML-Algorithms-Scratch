# Resources used
# 1. https://www.youtube.com/watch?v=KUrBgoIEQK0&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=5

# Notes:
# Does it qualify the list?
# 1. Non-linearity : Yes
# 2. Differentiable : Yes
# 3. Range : Yes
# 4. Monotonic : Yes
# 5. Computationaly efficient : Takes longer training time


# Softmax Activation Function
# Formula:
# f(x) = [e^(x)]/[sum(e^(x))] : x is a vector
# Mapping : (-inf, +inf) -->through e(x) (0, +inf) --> Normalized [0, 1]
# HyperParameter : No hyperparameter
# Use Cases: For output layers activation, multiclass classification
# Differentiated : 
# Drawback: [Computationally expensive as it is dependent on other o/ps]

import numpy as np

class SoftMax_function():
    
    def __init__(self):
        pass
    
    def activate(self, x):
        x = np.exp(x)
        print('Range b/w 0,inf with e(x): ',x)
        x = x/x.sum()
        print('Softmax value: ',x)
        print('Passes normalized check: ', (int(x.sum())==1))

softmax_function_1 = SoftMax_function()
softmax_function_1.activate(x=[32,8,6])
softmax_function_1.activate(x=[42,5,7])