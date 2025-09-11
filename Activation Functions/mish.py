# Resources used:
# 1. https://www.youtube.com/watch?v=DTVlyP-VihU&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=11

import numpy as np

x = np.random.random((1))
print(x)
o = x*np.tanh(np.log(1+np.exp(x)))

print(o)
