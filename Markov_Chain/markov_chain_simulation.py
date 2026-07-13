"""
Markov Chain Simulation
========================
A Markov Chain is a stochastic model describing a sequence of possible events
where the probability of each event depends only on the state attained in the
previous event (the "memoryless" property).

This module implements a MarkovChain class that can:
  1. Simulate a sequence of state transitions over N steps.
  2. Visualize the chain as a directed graph with weighted edges.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class MarkovChain:
    """
    Represents a discrete-time, finite-state Markov Chain.

    Attributes:
        states (list):            List of state names (e.g., ['Sunny', 'Cloudy', 'Rainy']).
        state_indices (dict):     Maps each state name to its integer index in the matrix.
        transition_matrix (ndarray): 2D array where entry [i][j] is the probability
                                     of moving from state i to state j. Each row sums to 1.
    """

    def __init__(self, states, transition_matrix):
        self.states = states

        # Build a lookup dict so we can convert a state name to its matrix row index.
        # e.g., {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
        self.state_indices = {state: i for i, state in enumerate(states)}

        # Convert the user-supplied list-of-lists into a NumPy array for fast math.
        self.transition_matrix = np.array(transition_matrix)

        # --- Validation ---
        # Every row of a transition matrix must sum to exactly 1 (total probability).
        # np.allclose handles floating-point rounding so 0.9999999 passes too.
        if not np.allclose(self.transition_matrix.sum(axis=1), 1):
            raise ValueError("Each row of the transition matrix must sum to 1.")

    def simulate(self, start_state, steps):
        """
        Simulates the Markov Chain for a given number of steps.

        Args:
            start_state (str): The initial state name (must be in self.states).
            steps (int):       Number of transitions to simulate.

        Returns:
            list: The sequence of visited states, length = steps + 1
                  (includes the starting state).
        """
        # Look up the matrix row index for the starting state.
        current_idx = self.state_indices[start_state]

        # Record the initial state as the first entry in the sequence.
        sequence = [start_state]

        for _ in range(steps):
            # Extract the probability distribution for the current state.
            # e.g., if current state is Sunny -> [0.6, 0.3, 0.1]
            probabilities = self.transition_matrix[current_idx]

            # np.random.choice picks an index from [0, 1, 2] weighted by probabilities.
            # This is the core stochastic step — the next state is randomly sampled.
            current_idx = np.random.choice(len(self.states), p=probabilities)

            # Append the human-readable state name to our running sequence.
            sequence.append(self.states[current_idx])

        return sequence

    def plot_graph(self):
        """
        Visualizes the Markov Chain as a directed graph.

        Nodes represent states; directed edges represent non-zero transition
        probabilities. Edge labels show the probability values. Self-loops
        (e.g., Sunny → Sunny) are drawn as curved arcs above/beside each node.
        """
        # --- Build the directed graph ---
        G = nx.DiGraph()

        # Add every non-zero transition as a weighted edge.
        # Rows are "from" states, columns are "to" states.
        for i, origin in enumerate(self.states):
            for j, destination in enumerate(self.states):
                weight = self.transition_matrix[i, j]
                if weight > 0:
                    G.add_edge(origin, destination, weight=weight)

        # --- Manual triangle layout ---
        # Instead of using spring/fruchterman layouts (which are non-deterministic),
        # we place nodes at fixed coordinates forming an equilateral triangle.
        # This ensures a clean, reproducible diagram.
        pos = {
            'Sunny': np.array([0.0, 0.866]),     # Top center
            'Cloudy': np.array([-1.0, -0.866]),  # Bottom left
            'Rainy': np.array([1.0, -0.866])     # Bottom right
        }

        plt.figure(figsize=(10, 8))

        # Draw the nodes as large light-blue circles with dark borders.
        nx.draw_networkx_nodes(
            G, pos,
            node_size=3000,
            node_color='#D6EAF8',      # Fill color
            edgecolors='#34495E',      # Border color
            linewidths=2
        )

        # Draw state name labels centered on each node.
        nx.draw_networkx_labels(
            G, pos,
            font_size=12,
            font_weight='bold',
            font_family='sans-serif'
        )

        ax = plt.gca()

        # --- Manually draw and label every edge ---
        # We don't use nx.draw_networkx_edges because we need fine-grained control
        # over curvature, label placement, and self-loop rendering.
        for u, v, d in G.edges(data=True):
            p1 = pos[u]  # Source node position
            p2 = pos[v]  # Destination node position
            weight_str = f"{d['weight']:.2f}"  # Format probability as "0.XX"

            if u == v:
                # ===== SELF-LOOP HANDLING =====
                # A self-loop is an arrow that leaves and returns to the same node.
                # We use a large arc radius (rad=3) to draw it outside the node circle.
                # The direction of the arc is chosen per-node so loops don't overlap.

                if u == 'Sunny':
                    # Loop arcs upward from the top node
                    connectionstyle = "arc3,rad=3"
                    text_pos = p1 + np.array([0.0, 0.25])       # Label above the node
                elif u == 'Cloudy':
                    # Loop arcs to the lower-left
                    connectionstyle = "arc3,rad=3"
                    text_pos = p1 + np.array([-0.22, -0.22])    # Label to the lower-left
                else:  # Rainy
                    # Use negative radius to flip arc to the right side
                    connectionstyle = "arc3,rad=-3"
                    text_pos = p1 + np.array([0.22, -0.22])     # Label to the lower-right

                # Draw the self-loop arrow.
                # shrinkA / shrinkB pull the arrow endpoints away from the node center
                # so the arrow doesn't overlap the node circle.
                ax.annotate(
                    "", xy=p1, xytext=p1,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#7F8C8D",
                        connectionstyle=connectionstyle,
                        linewidth=1.8,
                        mutation_scale=20,   # Arrowhead size
                        shrinkA=22,          # Shrink start point inward
                        shrinkB=22           # Shrink end point inward
                    )
                )

                # Place the probability label in a rounded box just outside the loop.
                ax.text(
                    text_pos[0], text_pos[1], weight_str,
                    color='#2C3E50',
                    fontweight='bold',
                    fontsize=10,
                    ha='center', va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",         # White background for readability
                        ec="#BDC3C7",       # Light gray border
                        alpha=0.9
                    )
                )

            else:
                # ===== INTER-STATE EDGE HANDLING =====
                # A gentle curve (rad=0.2) pushes the A→B arrow slightly to one side
                # so it doesn't sit directly on top of the B→A arrow.
                rad = 0.2
                ax.annotate(
                    "", xy=p2, xytext=p1,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#7F8C8D",
                        connectionstyle=f"arc3,rad={rad}",
                        linewidth=1.8,
                        mutation_scale=20,
                        shrinkA=25,   # Pull arrow start away from source node edge
                        shrinkB=25    # Pull arrow end away from destination node edge
                    )
                )

                # --- Label placement math ---
                # We want the probability label to sit at the midpoint of the arc,
                # offset perpendicular to the straight line between the two nodes.
                midpoint = (p1 + p2) / 2

                # Compute the perpendicular (normal) vector to the edge direction.
                direction = p2 - p1
                normal = np.array([-direction[1], direction[0]])  # 90-degree rotation
                normal = normal / np.linalg.norm(normal)          # Normalize to unit length

                # Push the label outward along the normal (away from the line)
                # and slightly along the edge direction for better centering.
                text_pos = midpoint + (normal * 0.18) + (direction * 0.05)

                # Draw the label in a rounded box on top of the arc.
                ax.text(
                    text_pos[0], text_pos[1], weight_str,
                    color='#2C3E50',
                    fontweight='bold',
                    fontsize=10,
                    ha='center', va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#BDC3C7",
                        alpha=0.9
                    )
                )

        plt.title(
            "Markov Chain State Transition Diagram",
            fontsize=14, fontweight='bold', pad=25
        )
        plt.axis('off')

        # Set axis limits slightly larger than the node positions so nothing
        # gets clipped at the edges (self-loops extend beyond the nodes).
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.4, 1.4)

        plt.tight_layout()
        plt.show()


# =============================================================================
# Execution Example
# =============================================================================
if __name__ == "__main__":
    # --- Define the system ---
    # Three weather states for our Markov Chain model.
    states = ['Sunny', 'Cloudy', 'Rainy']

    # --- Define the transition matrix ---
    # matrix[i][j] = probability of going from state i to state j.
    #
    #         To:  Sunny  Cloudy  Rainy
    # From:
    # Sunny       [ 0.6,   0.3,    0.1 ]   <- If Sunny today: 60% Sunny, 30% Cloudy, 10% Rainy
    # Cloudy      [ 0.2,   0.5,    0.3 ]   <- If Cloudy today: 20% Sunny, 50% Cloudy, 30% Rainy
    # Rainy       [ 0.1,   0.4,    0.5 ]   <- If Rainy today: 10% Sunny, 40% Cloudy, 50% Rainy
    matrix = [
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.4, 0.5]
    ]

    # Create the Markov Chain instance with our states and transition probabilities.
    mc = MarkovChain(states, matrix)

    # --- 1. Simulate weather ---
    # Generate a 10-day weather sequence starting from a Sunny day.
    # The sequence has 11 entries (day 0 + 10 transitions).
    history = mc.simulate(start_state='Sunny', steps=10)
    print("Simulated 10-day Weather Sequence:")
    print(" -> ".join(history))

    # --- 2. Visualize the chain ---
    # Draws the state transition diagram with labeled edges.
    mc.plot_graph()
