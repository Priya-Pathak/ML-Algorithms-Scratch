import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.state_indices = {state: i for i, state in enumerate(states)}
        self.transition_matrix = np.array(transition_matrix)
        
        # Validation check: rows must sum to approximately 1
        if not np.allclose(self.transition_matrix.sum(axis=1), 1):
            raise ValueError("Each row of the transition matrix must sum to 1.")

    def simulate(self, start_state, steps):
        """Simulates a sequence of states over N steps."""
        current_idx = self.state_indices[start_state]
        sequence = [start_state]
        
        for _ in range(steps):
            # Sample the next state index based on transition probabilities
            probabilities = self.transition_matrix[current_idx]
            current_idx = np.random.choice(len(self.states), p=probabilities)
            sequence.append(self.states[current_idx])
            
        return sequence

    def plot_graph(self):
        """Visualizes the Markov Chain as a directed graph."""
        G = nx.DiGraph()
        
        # Add nodes and weighted edges
        for i, origin in enumerate(self.states):
            for j, destination in enumerate(self.states):
                weight = self.transition_matrix[i, j]
                if weight > 0: # Only draw edges with non-zero probability
                    G.add_edge(origin, destination, weight=weight)
        
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))
        
        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Draw edges with curvature to handle bidirectional transitions
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, 
                               connectionstyle='arc3,rad=0.1', width=1.5, edge_color='gray')
        
        # Draw edge labels (probabilities)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=10)
        
        plt.title("Markov Chain State Transition Diagram", fontsize=14)
        plt.axis('off')
        plt.show()

# --- Execution Example ---
if __name__ == "__main__":
    # Define system states
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    # Define transition matrix
    # Row 1: Sunny  -> [Sunny, Cloudy, Rainy]
    # Row 2: Cloudy -> [Sunny, Cloudy, Rainy]
    # Row 3: Rainy  -> [Sunny, Cloudy, Rainy]
    matrix = [
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.4, 0.5]
    ]
    
    # Initialize Markov Chain
    mc = MarkovChain(states, matrix)
    
    # 1. Simulate a 10-day weather sequence starting from a Sunny day
    history = mc.simulate(start_state='Sunny', steps=10)
    print("Simulated 10-day Weather Sequence:")
    print(" -> ".join(history))
    
    # 2. Visualize the network structure
    mc.plot_graph()