# Imagine a world where only the MNIST dataset exists as images
# This plots the probability density P(x) of the MNIST dataset in a 2D latent space using t-SNE for dimensionality reduction.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns

# 1. Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Use t-SNE (Updated n_iter to max_iter)
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, method='exact')
X_embedded = tsne.fit_transform(X)

# Set up clean plotting styles
sns.set_theme(style='white')

# 3. Estimate and plot the probability density P(x)
plt.clf()
sns.kdeplot(
    x=X_embedded[:, 0], y=X_embedded[:, 1], 
    cmap="Purples", fill=True, thresh=0.05, alpha=0.3
)

# 4. Overlay the actual observed data points x
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1], 
    c=y, cmap='tab10', alpha=0.6, s=15, edgecolors='none'
)

plt.title("Visualizing $P(x)$ for a Digit World (t-SNE Projection)", fontsize=14, pad=15)
plt.xlabel("Latent Dimension 1", fontsize=10)
plt.ylabel("Latent Dimension 2", fontsize=10)

cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label("Digit Class (Modes of $P(x)$)", fontsize=10)
cbar.set_ticklabels([str(i) for i in range(10)])

plt.tight_layout()
plt.savefig("mnist_px_distribution.png", dpi=300)
print("Plot saved successfully as mnist_px_distribution.png")