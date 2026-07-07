import numpy as np
import matplotlib.pyplot as plt

# 1. Setup and Configurations
NUM_SIMULATIONS = 100_000  # Number of random scenarios to run
np.random.seed(42)         # Set seed for reproducible random numbers

print(f"Starting Monte Carlo simulation with {NUM_SIMULATIONS:,} trials...\n")

# 2. Define the Probability Distributions for each task (in days)
# Phase 1: Planning (Uniform distribution - equally likely between 5 and 10 days)
phase_1_simulations = np.random.uniform(low=5, high=10, size=NUM_SIMULATIONS)

# Phase 2: Coding (Normal distribution - average 15 days, standard deviation of 3 days)
phase_2_simulations = np.random.normal(loc=15, scale=3, size=NUM_SIMULATIONS)

# Phase 3: Testing (Triangular distribution - min 5 days, most likely 8 days, max 15 days)
phase_3_simulations = np.random.triangular(left=5, mode=8, right=15, size=NUM_SIMULATIONS)

# 3. Calculate the Aggregated Result for every single trial
# Total Timeline = Phase 1 + Phase 2 + Phase 3
total_timeline_simulations = phase_1_simulations + phase_2_simulations + phase_3_simulations

# 4. Statistical Analysis of the Results
mean_duration = np.mean(total_timeline_simulations)
std_duration = np.std(total_timeline_simulations)

# Calculate Risk Percentiles
p50 = np.percentile(total_timeline_simulations, 50)  # Median outcome
p70 = np.percentile(total_timeline_simulations, 70)  # 70% confidence level
p95 = np.percentile(total_timeline_simulations, 95)  # 95% confidence level (Worst-case limit)

# Display Results in Terminal
print("--- Simulation Results ---")
print(f"Average Project Duration : {mean_duration:.1f} days")
print(f"Standard Deviation       : {std_duration:.1f} days")
print(f"50% Confidence (P50)     : Project finished by day {p50:.1f}")
print(f"70% Confidence (P70)     : Project finished by day {p70:.1f}")
print(f"95% Confidence (P95)     : Project finished by day {p95:.1f} (Highly conservative)")

# 5. Data Visualization (Plotting the Output Histogram)
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(
    total_timeline_simulations, 
    bins=100, 
    density=True, 
    alpha=0.6, 
    color='royalblue', 
    edgecolor='black'
)

# Add visualization lines for percentiles
plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_duration:.1f} days)')
plt.axvline(p70, color='orange', linestyle='dashed', linewidth=2, label=f'70% Confidence ({p70:.1f} days)')
plt.axvline(p95, color='darkred', linestyle='dashed', linewidth=2, label=f'95% Confidence ({p95:.1f} days)')

# Styling the Plot
plt.title(f'Monte Carlo Simulation: Total Project Timeline ({NUM_SIMULATIONS:,} Trials)', fontsize=14)
plt.xlabel('Total Days to Complete Project', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='upper right', fontsize=11)

# Show the plot
plt.tight_layout()
plt.show()