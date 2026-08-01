import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Define the Autoencoder Architecture
class RepresentationAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=4):
        super(RepresentationAutoencoder, self).__init__()
        # Encoder: Compresses 64 features -> 32 -> 16 -> 4 (Latent Space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim) # Latent bottleneck
        )
        # Decoder: Reconstructs 4 -> 16 -> 32 -> 64 (Output Space)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Bounds outputs between 0 and 1
        )

    def forward(self, x):
        # We write explicit forward steps to extract intermediate representations
        enc_layer1 = self.encoder[0](x)
        enc_layer1_act = self.encoder[1](enc_layer1)
        enc_layer2 = self.encoder[2](enc_layer1_act)
        enc_layer2_act = self.encoder[3](enc_layer2)
        latent_space = self.encoder[4](enc_layer2_act)
        
        dec_layer1 = self.decoder[0](latent_space)
        dec_layer1_act = self.decoder[1](dec_layer1)
        dec_layer2 = self.decoder[2](dec_layer1_act)
        dec_layer2_act = self.decoder[3](dec_layer2)
        output_space = self.decoder[4](dec_layer2_act)
        
        # Return everything so we can map it
        return {
            "Input": x,
            "Encoder Layer 1": enc_layer1_act,
            "Encoder Layer 2": enc_layer2_act,
            "Latent Space": latent_space,
            "Decoder Layer 1": dec_layer1_act,
            "Decoder Layer 2": dec_layer2_act,
            "Decoder Output": output_space
        }

# 2. Initialize Network and Generate Dummy Batch Data
# Setting seed for reproducible random weight activations
torch.manual_seed(42)
model = RepresentationAutoencoder(input_dim=64, latent_dim=4)
model.eval() # Evaluation mode (disables dropout if any)

# Simulate a batch of 8 input samples, each with 64 features
dummy_input = torch.rand(8, 64)

# 3. Extract Representations
with torch.no_grad():
    representations = model(dummy_input)

# 4. Plot the Representation Spaces as Heatmaps
fig, axes = plt.subplots(1, 5, figsize=(20, 6), gridspec_kw={'width_ratios': [4, 2, 1, 2, 4]})
cmap_style = "viridis"

# Plot 1: Input Space (8 samples x 64 features)
sns.heatmap(representations["Input"].numpy(), ax=axes[0], cmap=cmap_style, cbar=False)
axes[0].set_title("Input Space\n(8 x 64)", fontsize=12)
axes[0].set_ylabel("Batch Samples")
axes[0].set_xlabel("Features")

# Plot 2: Intermediate Encoder Representation (8 samples x 16 features)
sns.heatmap(representations["Encoder Layer 2"].numpy(), ax=axes[1], cmap=cmap_style, cbar=False)
axes[1].set_title("Encoder Space\n(8 x 16)", fontsize=12)
axes[1].set_xlabel("Hidden Neurons")

# Plot 3: Latent Bottleneck Core (8 samples x 4 features)
sns.heatmap(representations["Latent Space"].numpy(), ax=axes[2], cmap="magma", cbar=False)
axes[2].set_title("Latent Space\n(8 x 4)", fontsize=12, color="red", fontweight="bold")
axes[2].set_xlabel("Latent Dim")

# Plot 4: Intermediate Decoder Representation (8 samples x 16 features)
sns.heatmap(representations["Decoder Layer 1"].numpy(), ax=axes[3], cmap=cmap_style, cbar=False)
axes[3].set_title("Decoder Space\n(8 x 16)", fontsize=12)
axes[3].set_xlabel("Hidden Neurons")

# Plot 5: Final Reconstructed Output Space (8 samples x 64 features)
sns.heatmap(representations["Decoder Output"].numpy(), ax=axes[4], cmap=cmap_style, cbar=False)
axes[4].set_title("Reconstructed Output\n(8 x 64)", fontsize=12)
axes[4].set_xlabel("Features")

plt.tight_layout()
plt.show()
