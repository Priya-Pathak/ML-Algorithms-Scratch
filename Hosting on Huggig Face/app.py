"""
VAE Latent Space Explorer — Gradio app for Hugging Face Spaces.

Visualizes the latent space of a Variational Autoencoder trained on MNIST.
Users can click any point on the 2D t-SNE scatter plot to see the decoded
image from that region of latent space.

Usage:
    python app.py          # runs locally
    # or deploy to HF Spaces by uploading app.py + requirements.txt
"""

import os
import urllib.request
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px

import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# VAE Model (embedded from VAE/VAE/vae_mnist.py)
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    MLP Variational Autoencoder for MNIST.

    Architecture:
        Encoder: Linear(784, 500) -> ReLU -> mu/logvar heads
        Latent:  20-dim
        Decoder: Linear(20, 500) -> ReLU -> Linear(500, 784) -> Sigmoid
    """

    def __init__(self, input_dim=784, hidden_dim=500, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# ---------------------------------------------------------------------------
# Weight loading — downloads best_model.pt from GitHub if not cached locally.
# Falls back to a local path (the original VAE repo) for local development.
# ---------------------------------------------------------------------------

WEIGHTS_URL = (
    "https://raw.githubusercontent.com/Priya-Pathak/VAE/main/"
    "results/mnist/best_model.pt"
)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best_model.pt")

# Local fallback for development — points to the trained model on disk.
LOCAL_WEIGHTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "VAE", "VAE", "results", "mnist", "best_model.pt")
)


def load_weights() -> str:
    """Load model weights: cache dir > GitHub download > local fallback."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH

    # Try downloading from GitHub (works on HF Spaces where local fallback doesn't exist).
    try:
        print(f"Downloading model weights from {WEIGHTS_URL} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("Download complete.")
        return WEIGHTS_PATH
    except Exception as e:
        print(f"GitHub download failed ({e}), trying local path ...")

    # Local fallback for development.
    if os.path.exists(LOCAL_WEIGHTS):
        import shutil
        shutil.copy2(LOCAL_WEIGHTS, WEIGHTS_PATH)
        print(f"Copied weights from {LOCAL_WEIGHTS}")
        return WEIGHTS_PATH

    raise FileNotFoundError(
        f"Could not find model weights. Tried:\n"
        f"  1. {WEIGHTS_PATH} (cache)\n"
        f"  2. {WEIGHTS_URL} (GitHub)\n"
        f"  3. {LOCAL_WEIGHTS} (local)"
    )


# ---------------------------------------------------------------------------
# Pre-computation — runs once at startup
# ---------------------------------------------------------------------------

LATENT_DIM = 20
DEVICE = "cpu"


def prepare_latent_space():
    """
    Load model + MNIST test set, encode every test image, run t-SNE,
    and build a KDTree for fast nearest-neighbour lookup on click.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 2D scatter of the latent space.
    tree : scipy.spatial.KDTree
        Nearest-neighbour index over the 2D t-SNE coordinates.
    z_2d : np.ndarray  (10000, 2)
        t-SNE coordinates (kept for tree construction).
    z_mu : np.ndarray  (10000, 20)
        Raw encoder means (needed for decoding).
    images : np.ndarray (10000, 28, 28)
        Original test images.
    labels : np.ndarray (10000,)
        Digit labels.
    """
    # --- Model ---
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    weights_path = load_weights()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # --- MNIST test set ---
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), "data"),
        train=False,
        download=True,
        transform=transform,
    )

    all_mu = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for x, y in test_dataset:
            flat = x.view(1, -1).to(DEVICE)
            mu, _ = model.encode(flat)
            all_mu.append(mu.squeeze(0).cpu().numpy())
            all_labels.append(y)
            all_images.append(x.squeeze(0).numpy())

    z_mu = np.stack(all_mu)          # (10000, 20)
    labels = np.array(all_labels)    # (10000,)
    images = np.stack(all_images)    # (10000, 28, 28)

    # --- t-SNE reduction to 2D ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_mu)  # (10000, 2)

    # --- KDTree for click lookup ---
    tree = KDTree(z_2d)

    # --- Plotly scatter ---
    df = pd.DataFrame({
        "x": z_2d[:, 0],
        "y": z_2d[:, 1],
        "digit": labels.astype(str),
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="digit",
        title="MNIST VAE Latent Space (20D → t-SNE 2D)",
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "digit": "Digit"},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
        width=900,
        height=700,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="white")))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Digit",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig, tree, z_2d, z_mu, images, labels, model


print("Preparing latent space (this runs once at startup)...")
FIG, TREE, Z_2D, Z_MU, IMAGES, LABELS, MODEL = prepare_latent_space()
print("Ready.")


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def decode_from_coords(tsne_x, tsne_y):
    """Decode a point from t-SNE coordinates entered by the user."""
    if tsne_x is None or tsne_y is None:
        return None, None, "Enter t-SNE coordinates and click Decode."

    _, idx = TREE.query([tsne_x, tsne_y])

    with torch.no_grad():
        z = torch.tensor(Z_MU[idx : idx + 1], dtype=torch.float32, device=DEVICE)
        recon = MODEL.decode(z).squeeze().numpy()

    recon_img = (recon.reshape(28, 28) * 255).astype(np.uint8)
    orig_img = (IMAGES[idx] * 255).astype(np.uint8)

    return orig_img, recon_img, f"Nearest sample index: {idx} | True digit: {LABELS[idx]}"


def decode_random():
    """Decode a random point sampled from the latent space."""
    idx = np.random.randint(len(Z_MU))

    with torch.no_grad():
        z = torch.tensor(Z_MU[idx : idx + 1], dtype=torch.float32, device=DEVICE)
        recon = MODEL.decode(z).squeeze().numpy()

    recon_img = (recon.reshape(28, 28) * 255).astype(np.uint8)
    orig_img = (IMAGES[idx] * 255).astype(np.uint8)

    return orig_img, recon_img, f"Random sample index: {idx} | True digit: {LABELS[idx]}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="VAE Latent Space Explorer") as demo:
    gr.Markdown(
        "# VAE Latent Space Explorer\n"
        "An MLP Variational Autoencoder trained on MNIST compresses each "
        "28\u00d728 digit image into a 20-dimensional latent vector. "
        "t-SNE projects it to 2D for visualization.\n\n"
        "Enter t-SNE coordinates (or click **Random Sample**) to see the "
        "decoded image from that region of latent space."
    )

    with gr.Row():
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Latent Space")
        with gr.Column(scale=2):
            gr.Markdown("### Decode from Latent Space")
            with gr.Row():
                tsne_x = gr.Number(label="t-SNE 1", value=0.0)
                tsne_y = gr.Number(label="t-SNE 2", value=0.0)
            with gr.Row():
                decode_btn = gr.Button("Decode", variant="primary")
                random_btn = gr.Button("Random Sample")
            orig_image = gr.Image(label="Original", height=250)
            recon_image = gr.Image(label="Decoded", height=250)
            info_text = gr.Textbox(label="Info", interactive=False)

    # Load the plot on startup
    demo.load(fn=lambda: FIG, outputs=plot_output)

    # Decode from user-entered coordinates
    decode_btn.click(
        fn=decode_from_coords,
        inputs=[tsne_x, tsne_y],
        outputs=[orig_image, recon_image, info_text],
    )

    # Decode a random sample
    random_btn.click(
        fn=decode_random,
        outputs=[orig_image, recon_image, info_text],
    )

if __name__ == "__main__":
    demo.launch()
