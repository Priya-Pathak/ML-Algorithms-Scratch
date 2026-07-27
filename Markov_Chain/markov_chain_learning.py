import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define states for readability
states = ["Novice", "Intermediate", "Competent", "Advanced", "Expert"]

# Transition Probability Matrix (P)
P = np.array([
    [0.60, 0.40, 0.00, 0.00, 0.00],  # Novice
    [0.15, 0.55, 0.30, 0.00, 0.00],  # Intermediate
    [0.00, 0.15, 0.55, 0.30, 0.00],  # Competent
    [0.00, 0.00, 0.20, 0.55, 0.25],  # Advanced
    [0.00, 0.00, 0.00, 0.10, 0.90]   # Expert
])

# ==========================================
# 1. GENERATE STUDENT SIMULATION DATA
# ==========================================
np.random.seed(42)
current_idx = 0  # Start as Novice
weeks = 52
timeline = list(range(weeks + 1))
numerical_history = [current_idx]

for week in range(weeks):
    current_idx = np.random.choice(len(states), p=P[current_idx])
    numerical_history.append(current_idx)

# ==========================================
# 2. RENDER THE PROFESSIONAL PLOTS
# ==========================================
# Create a crisp 2-panel dashboard layout
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.3]})
sns.set_theme(style="whitegrid")

# --- Left Panel: Transition Probability Matrix Heatmap ---
sns.heatmap(
    P, 
    annot=True, 
    fmt=".2f", 
    cmap="Blues", 
    xticklabels=states, 
    yticklabels=states, 
    cbar=False, 
    linewidths=1.5, 
    linecolor="white",
    annot_kws={"size": 11, "weight": "bold"},
    ax=axes[0]
)
axes[0].set_title("Transition Flow Matrix\n(Row = Current State ➔ Column = Next State)", fontsize=13, pad=15, weight="bold")
axes[0].set_ylabel("Current State", fontsize=11, weight="bold")
axes[0].set_xlabel("Next State", fontsize=11, weight="bold")
axes[0].tick_params(axis='both', which='major', labelsize=10)

# --- Right Panel: 1-Year Student Trajectory Path ---
axes[1].plot(timeline, numerical_history, color="#0d6efd", linewidth=2.5, marker='o', markersize=4, label="Student Journey")

# Enhance the step-by-step layout
axes[1].set_title("Simulated Student Progress Journey Over 1 Year", fontsize=13, pad=15, weight="bold")
axes[1].set_xlabel("Weeks elapsed", fontsize=11, weight="bold")
axes[1].set_ylabel("Intelligence tier achieved", fontsize=11, weight="bold")
axes[1].set_yticks(range(len(states)))
axes[1].set_yticklabels(states, fontsize=10, weight="bold")
axes[1].set_xlim(0, weeks)
axes[1].set_ylim(-0.5, len(states) - 0.5)

# Highlight milestones (e.g. hitting Expert tier)
expert_weeks = [w for w, s in enumerate(numerical_history) if s == 4]
if expert_weeks:
    first_expert_week = expert_weeks[0]
    axes[1].axvline(x=first_expert_week, color="#198754", linestyle="--", linewidth=1.5)
    axes[1].text(first_expert_week + 0.5, 3.8, f"Hit Expert at Week {first_expert_week}", color="#198754", weight="bold", fontsize=10)

plt.tight_layout()
plt.show()
