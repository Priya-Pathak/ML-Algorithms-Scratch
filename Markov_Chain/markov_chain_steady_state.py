import numpy as np

# 1. Define the Transition Matrix P
# Example: 2 states (e.g., State 0: Sunny, State 1: Rainy)
# Row 0: From Sunny -> [0.8 Sunny, 0.2 Rainy]
# Row 1: From Rainy -> [0.4 Sunny, 0.6 Rainy]
P = np.array([
    [0.8, 0.2],
    [0.4, 0.6]
])

# --- METHOD 1: Iterative Simulation ---
# Start with ANY initial probability distribution (e.g., 100% chance of Sunny)
state = np.array([1.0, 0.0])

print("--- Iterative Approach ---")
print(f"Step 0: {state}")

for step in range(1, 15):
    state = np.dot(state, P)
    if step in [1, 2, 3, 5, 10, 14]:
        print(f"Step {step:2d}: {np.round(state, 5)}")

# --- METHOD 2: Exact Algebraic Solution ---
# We solve (P^T - I) * pi^T = 0, with constraint sum(pi) = 1
# This is equivalent to finding the eigenvector corresponding to eigenvalue = 1
eigenvalues, eigenvectors = np.linalg.eig(P.T)

# Find index where eigenvalue is approximately 1
index = np.argmin(np.abs(eigenvalues - 1.0))
steady_state_raw = eigenvectors[:, index].real

# Normalize so the probabilities sum to 1
steady_state_exact = steady_state_raw / np.sum(steady_state_raw)

print("\n--- Exact Solution ---")
print(f"Steady State: {np.round(steady_state_exact, 5)}")