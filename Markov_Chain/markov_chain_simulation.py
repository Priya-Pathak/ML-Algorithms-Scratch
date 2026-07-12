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
        """Visualizes the Markov Chain with guaranteed zero overlapping text and clear arrows."""
        G = nx.DiGraph()
        for i, origin in enumerate(self.states):
            for j, destination in enumerate(self.states):
                weight = self.transition_matrix[i, j]
                if weight > 0:
                    G.add_edge(origin, destination, weight=weight)
        
        # 1. Fix nodes in a rigid triangle layout with plenty of breathing room
        # We manually space out the coordinates (x, y) to maximize distance
        pos = {
            'Sunny': np.array([0.0, 0.866]),   # Top center
            'Cloudy': np.array([-1.0, -0.866]), # Bottom left
            'Rainy': np.array([1.0, -0.866])    # Bottom right
        }
        
        plt.figure(figsize=(10, 8))
        
        # Draw background base nodes cleanly
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#D6EAF8', edgecolors='#34495E', linewidths=2)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_family='sans-serif')
        
        ax = plt.gca()
        
        # 2. Manually draw and label edges to handle precise offsets
        for u, v, d in G.edges(data=True):
            p1 = pos[u]
            p2 = pos[v]
            weight_str = f"{d['weight']:.2f}"
            
            if u == v:
                # --- SELF LOOP HANDLING ---
                # Determine loop direction based on where the node sits in the triangle
                if u == 'Sunny':
                    connectionstyle = "arc3,rad=3"
                    text_pos = p1 + np.array([0.0, 0.25])
                elif u == 'Cloudy':
                    connectionstyle = "arc3,rad=3"
                    text_pos = p1 + np.array([-0.22, -0.22])
                else:  # Rainy
                    connectionstyle = "arc3,rad=-3"
                    text_pos = p1 + np.array([0.22, -0.22])
                
                # Draw the loop arrow
                ax.annotate("", xy=p1, xytext=p1,
                            arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", connectionstyle=connectionstyle,
                                            linewidth=1.8, mutation_scale=20, shrinkA=22, shrinkB=22))
                
                # Place the text centered just outside the loop
                ax.text(text_pos[0], text_pos[1], weight_str, color='#2C3E50', 
                        fontweight='bold', fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.9))
                
            else:
                # --- INTER-STATE EDGES ---
                # A slight curve pushes A->B away from B->A
                rad = 0.2
                ax.annotate("", xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", connectionstyle=f"arc3,rad={rad}",
                                            linewidth=1.8, mutation_scale=20, shrinkA=25, shrinkB=25))
                
                # Calculate the exact geometric midpoint of the arc path to place the label
                midpoint = (p1 + p2) / 2
                direction = p2 - p1
                normal = np.array([-direction[1], direction[0]])  # Perpendicular vector
                normal = normal / np.linalg.norm(normal)
                
                # Push the text block outward away from the direct line path so it sits outside the arc
                text_pos = midpoint + (normal * 0.18) + (direction * 0.05)
                
                ax.text(text_pos[0], text_pos[1], weight_str, color='#2C3E50', 
                        fontweight='bold', fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.9))
        
        plt.title("Markov Chain State Transition Diagram", fontsize=14, fontweight='bold', pad=25)
        plt.axis('off')
        
        # Expand limits so elements close to edges don't get chopped off
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.4, 1.4)
        
        plt.tight_layout()
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