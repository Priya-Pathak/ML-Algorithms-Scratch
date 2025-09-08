# Resources used:
# 1. https://www.youtube.com/watch?v=DTVlyP-VihU&list=PL1u-h-YIOL0vOwds4QCAco2KMeOt7_zSh&index=8

import numpy as np

x = np.random.random((2,5))
print(x.shape)

# 12 hidden neurons
w = np.random.random((12,5))
b = np.random.random((12))
print(w.shape)
print(x.shape)

z = np.dot(x, w.T)+b
print(z.shape)

z = z.reshape((2,4,3))
o = np.max(z, axis = 2)
print(o)
print(o.shape)