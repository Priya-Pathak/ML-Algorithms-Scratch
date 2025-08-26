# Resources used
# 1. https://www.youtube.com/watch?v=l4intEWyMlo&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=2

# Notes:
# Does it qualify the list?
# 1. Non-linearity : Yes
# 2. Differentiable : Yes
# 3. Range : Yes
# 4. Monotonic : Yes
# 5. Computationaly efficient : Takes longer training time


# Step Activation Function
# Formula:
# f(x) = 1 if x => theta else 0
# Mapping : (-inf, +inf) --> 0 or 1
# HyperParameter : theta
# Use Cases: Binary classification
# Drawback: [0 derivative, Only for binary classification, Hard threshold]

class Step_function():
    
    def __init__(self, theta= 0):
        self.theta = theta
    
    def activate(self, x):
        if x < self.theta:
            return 0
        else:
            return 1

step_function_5 = Step_function(theta=5)
print('x=32 and theta=5: ',step_function_5.activate(x=32))
print('x=-8 and theta=-8: ',step_function_5.activate(x=-8))

step_function_0 = Step_function(theta=0)
print('x=9 and theta=0: ',step_function_0.activate(x=9))
print('x=-92 and theta=0: ',step_function_0.activate(x=-92))