# This is for Law of Unconcious Expectation (LOTUS) demonstration with a discrete random variable.
import numpy as np

# 1. Define discrete Random Variable X and its PMF
x_values = np.array([-2, -1, 0, 1, 2])
p_x = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

# Function g(x)
def g(x):
    return x ** 2

# --- Method 1: LOTUS (Direct Calculation) ---
# E[g(X)] = sum( g(x) * P(X = x) )
lotus_expectation = np.sum(g(x_values) * p_x)

# --- Method 2: Traditional Method (Deriving PMF of Y = g(X) first) ---
y_raw = g(x_values)
y_values = np.unique(y_raw)
p_y = np.array([p_x[y_raw == val].sum() for val in y_values])

traditional_expectation = np.sum(y_values * p_y)

# --- Method 3: Monte Carlo Simulation ---
np.random.seed(42)
samples_x = np.random.choice(x_values, size=1_000_000, p=p_x)
monte_carlo_expectation = np.mean(g(samples_x))

print(f"LOTUS Expectation:       {lotus_expectation:.4f}")
print(f"Traditional Expectation: {traditional_expectation:.4f}")
print(f"Monte Carlo Mean:        {monte_carlo_expectation:.4f}")