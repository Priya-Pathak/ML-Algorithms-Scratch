# Resources used:
# 1. https://www.youtube.com/watch?v=DTVlyP-VihU&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=10

import numpy as np

x = np.random.random((1))
print(x.shape)
o = x*(1/(1+np.exp(x)))

print(o)
