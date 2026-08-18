import numpy as np
import matplotlib.pyplot as plt

# 1. Target unnormalized function f(x)
def f(x):
    # A asymmetric two-peak function
    return np.exp(-0.5 * (x + 2)**2) + 1.8 * np.exp(-0.5 * (x - 2)**2 / 0.6)

# 2. Pure Metropolis MCMC (Simple symmetric random walk)
def simple_metropolis(target_func, num_samples=60000, step_size=1.5, x_init=0.0):
    samples = []
    x_curr = x_init
    
    for _ in range(num_samples):
        # Uniform symmetric proposal: x_proposed = x_curr + random_step in [-step_size, step_size]
        x_prop = x_curr + np.random.uniform(-step_size, step_size)
        
        # Pure Metropolis ratio (no Hastings proposal correction needed)
        ratio = target_func(x_prop) / target_func(x_curr)
        alpha = min(1.0, ratio)
        
        # Accept or stay
        if np.random.rand() < alpha:
            x_curr = x_prop
            
        samples.append(x_curr)
        
    return np.array(samples)

# --- Run MCMC Simulation ---
num_samples = 60000
burn_in = 10000

samples = simple_metropolis(f, num_samples=num_samples, step_size=1.5, x_init=0.0)
valid_samples = samples[burn_in:]

# --- Compute Exact Normalized PDF for Visual Comparison ---
x_grid = np.linspace(-6, 6, 1000)
actual_unnormalized = f(x_grid)
total_area = np.trapz(actual_unnormalized, x_grid)  # Area under f(x)
actual_normalized_pdf = actual_unnormalized / total_area

# --- Plotting ---
plt.figure(figsize=(12, 5))

# Plot 1: Normalized Probability Density (MCMC Histogram vs True PDF)
plt.subplot(1, 2, 1)
plt.hist(valid_samples, bins=70, density=True, alpha=0.5, color='crimson', label='MCMC Samples (Normalized)')
plt.plot(x_grid, actual_normalized_pdf, 'k-', lw=2.5, label='Actual Normalized PDF')
plt.title('1. Density Comparison (Normalized)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Direct Function Shape (Re-scaled MCMC vs Unnormalized f(x))
plt.subplot(1, 2, 2)
counts, bin_edges = np.histogram(valid_samples, bins=70, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
mcmc_unnormalized_estimate = counts * total_area

plt.plot(x_grid, actual_unnormalized, 'b-', lw=2, label='Actual Function $f(x)$')
plt.plot(bin_centers, mcmc_unnormalized_estimate, 'r.', ms=5, label='MCMC Estimate Points')
plt.title('2. Actual $f(x)$ vs MCMC Reconstruction')
plt.xlabel('x')
plt.ylabel('$f(x)$ Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()