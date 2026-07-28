import numpy as np

# 1. Define the 5x5 transition probability matrix (P)
# Rows & Columns map to: [S0:Novice, S1:Competent, S2:Proficient, S3:Advanced, S4:Expert]
P = np.array([
    [0.30, 0.60, 0.10, 0.00, 0.00],  # S0 transitions
    [0.15, 0.35, 0.45, 0.05, 0.00],  # S1 transitions
    [0.00, 0.15, 0.40, 0.40, 0.05],  # S2 transitions
    [0.00, 0.00, 0.20, 0.50, 0.30],  # S3 transitions
    [0.00, 0.00, 0.00, 0.25, 0.75]   # S4 transitions
])

def get_steady_state_algebraic(matrix):
    """
    Solves the linear system equations: pi * P = pi and sum(pi) = 1
    Formulated as: (P^T - I) * pi = 0
    """
    n = matrix.shape[0]
    # Set up the system matrix (Transposed P minus Identity matrix)
    A = matrix.T - np.eye(n)
    
    # Replace the last equation row with the constraint that all probabilities sum to 1
    A[-1] = np.ones(n)
    
    # Target vector (zeros, with the last element as 1 for the sum constraint)
    b = np.zeros(n)
    b[-1] = 1.0
    
    # Solve the linear system
    return np.linalg.solve(A, b)

def get_steady_state_iterative(matrix, iterations=100):
    """
    Simulates a population starting entirely as Novices (S0)
    and multiplies by the transition matrix repeatedly.
    """
    # Start vector: 100% of the population starts at S0 (Novice)
    state_vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    for _ in range(iterations):
        state_vector = np.dot(state_vector, matrix)
        
    return state_vector

# --- Execution ---
states = ["S0 (Novice)", "S1 (Competent)", "S2 (Proficient)", "S3 (Advanced)", "S4 (Expert)"]

exact_solution = get_steady_state_algebraic(P)
iterative_solution = get_steady_state_iterative(P, iterations=100)

print("=== COGNITIVE MODEL STEADY-STATE DISTRIBUTION ===")
header = f"{'State':<16} | {'Exact Prob':<12} | {'Estimate %':<11} | {'Iterative Prob':<15} | {'Iterative %'}"
print(header)
print("-" * len(header))
for i, state in enumerate(states):
    ep = f"{exact_solution[i]*100:.2f}%"
    ip = f"{iterative_solution[i]*100:.2f}%"
    print(f"{state:<16} | {exact_solution[i]:<12.6f} | {ep:<10} | {iterative_solution[i]:<14.6f} | {ip}")

print("\n[Verification] Iterative simulation check after 100 steps:")
print(np.round(iterative_solution, 6))