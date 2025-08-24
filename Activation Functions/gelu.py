# Formula: 
# GELU(x)=x⋅Φ(x)
# Where Φ(x)
# Φ(x) is the cumulative distribution function (CDF) of the standard normal (Gaussian) distribution.

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
