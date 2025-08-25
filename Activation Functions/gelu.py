# Video reference : https://www.youtube.com/watch?v=kMpptn-6jaw
# Formula: 
# GELU(x)=x⋅Φ(x)
# Where Φ(x(i,j)) = P(X(i,j)<=x(i,j)) [Implemented using Gauss Error Function]
# Φ(x) is the cumulative distribution function (CDF) of the standard normal (Gaussian) distribution.
# Gelu combines the RELU and dropout in deterministic way
# All the i/p's x fed to the neuron before activation have show normal distribution with N(0,1)
# ReLU is zero to identity mapping, [x<0 -> 0, x>0 ->x] 

import numpy as np

def gelu(x, approximate=False):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    Args:
        x: Input numpy array or scalar.
        approximate: If True, use the tanh-based approximation (faster).
    Returns:
        GELU-activated values (same shape as x).
    """
    if approximate:
        # tanh-based approximation (Hendrycks & Gimpel, 2016)
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    else:
        # Exact formula using the error function
        from scipy.special import erf
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# Example usage:
x = np.linspace(-3, 3, 7)
print("Exact GELU:", gelu(x))
print("Approximate GELU:", gelu(x, approximate=True))
