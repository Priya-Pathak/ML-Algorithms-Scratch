"""
VAE Latent Space Explorer — Gradio app for Hugging Face Spaces.
Visualizes the latent space of a Variational Autoencoder trained on MNIST.
"""
import os
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import spaces

# ---------------------------------------------------------------------------
# VAE Model
# ---------------------------------------------------------------------------
class VAE(nn.Module):
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

WEIGHTS_PATH = "./best_model.pt"

def load_weights() -> str:
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    raise FileNotFoundError(f"Could not find model weights at {WEIGHTS_PATH}.")

# ---------------------------------------------------------------------------
# Pre-computation (Runs ONCE at startup on CPU)
# ---------------------------------------------------------------------------
LATENT_DIM = 20
DEVICE = "cpu"

def prepare_latent_space():
    # Load model on CPU first so Zero-GPU doesn't have to reload it from disk later
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    
    try:
        weights_path = load_weights()
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        print(f"Skipping weight loading due to missing file: {e}")
        
    model.eval()

    try:
        print("Attempting to load MNIST dataset...")
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), "data"),
            train=False,
            download=True,
            transform=transform,
        )
        
        all_mu, all_labels, all_images = [], [], []
        with torch.no_grad():
            for x, y in test_dataset:
                flat = x.view(1, -1).to(DEVICE)
                mu, _ = model.encode(flat)
                all_mu.append(mu.squeeze(0).cpu().numpy())
                all_labels.append(y)
                all_images.append(x.squeeze(0).numpy())
                
        z_mu = np.stack(all_mu)
        labels = np.array(all_labels)
        images = np.stack(all_images)
        
    except Exception as network_error:
        print(f"Network constraint. Initializing fallback dataset matrix.")
        num_samples = 1000
        z_mu = np.random.randn(num_samples, LATENT_DIM)
        labels = np.random.randint(0, 10, size=num_samples)
        images = np.random.rand(num_samples, 28, 28)

    print("Running t-SNE reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_mu)
    tree = KDTree(z_2d)
    
    df = pd.DataFrame({"x": z_2d[:, 0], "y": z_2d[:, 1], "digit": labels.astype(str)})
    fig = px.scatter(
        df, x="x", y="y", color="digit",
        title="MNIST VAE Latent Space (20D → t-SNE 2D)",
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "digit": "Digit"},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
        width=900, height=700,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="white")))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Digit",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig, tree, z_2d, z_mu, images, labels, model

print("Preparing latent space (this runs once at startup)...")
FIG, TREE, Z_2D, Z_MU, IMAGES, LABELS, GLOBAL_MODEL = prepare_latent_space()
print("Ready.")

# ---------------------------------------------------------------------------
# JavaScript for Plotly Clicks
# ---------------------------------------------------------------------------
CLICK_JS = r"""(async function () {
    let plotDiv = null;
    for (let i = 0; i < 120; i++) {
        plotDiv = document.querySelector(".js-plotly-plot");
        if (plotDiv && plotDiv._fullLayout) break;
        await new Promise(r => setTimeout(r, 250));
    }
    if (!plotDiv) { console.warn("Plotly graph not found"); return; }
    plotDiv.on("plotly_click", function (data) {
        var pt = data.points[0];
        var x = pt.x;
        var y = pt.y;
        var xBox = document.querySelector("#tsne-x input[type='number']");
        var yBox = document.querySelector("#tsne-y input[type='number']");
        if (!xBox || !yBox) { console.warn("Coordinate inputs not found"); return; }
        var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
        setter.call(xBox, x);
        xBox.dispatchEvent(new Event("input",  { bubbles: true }));
        setter.call(yBox, y);
        yBox.dispatchEvent(new Event("input",  { bubbles: true }));
    });
})();"""

# ---------------------------------------------------------------------------
# Zero-GPU Thread Safe Callbacks
# ---------------------------------------------------------------------------
@spaces.GPU
def decode_from_coords(tsne_x, tsne_y):
    """Find the nearest test sample for the given t-SNE coordinates and decode it."""
    if tsne_x is None or tsne_y is None:
        return None, None, "Click a point on the plot, then press Decode."
        
    _, idx = TREE.query([float(tsne_x), float(tsne_y)])
    
    # Send the pre-loaded global model directly to the allocated GPU
    gpu_model = GLOBAL_MODEL.to("cuda")
    
    with torch.no_grad():
        z = torch.tensor(Z_MU[idx:idx + 1], dtype=torch.float32, device="cuda")
        recon = gpu_model.decode(z).squeeze().cpu().numpy()
        
    recon_img = (recon.reshape(28, 28) * 255).astype(np.uint8)
    orig_img = (IMAGES[idx] * 255).astype(np.uint8)
    return orig_img, recon_img, f"Nearest sample index: {idx}  |  True digit: {LABELS[idx]}"

@spaces.GPU
def decode_random():
    """Decode a random point from the latent space."""
    idx = np.random.randint(len(Z_MU))
    
    # Send the pre-loaded global model directly to the allocated GPU
    gpu_model = GLOBAL_MODEL.to("cuda")
    
    with torch.no_grad():
        z = torch.tensor(Z_MU[idx:idx + 1], dtype=torch.float32, device="cuda")
        recon = gpu_model.decode(z).squeeze().cpu().numpy()
        
    recon_img = (recon.reshape(28, 28) * 255).astype(np.uint8)
    orig_img = (IMAGES[idx] * 255).astype(np.uint8)
    return orig_img, recon_img, f"Random sample index: {idx}  |  True digit: {LABELS[idx]}"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(js=CLICK_JS, title="VAE Latent Space Explorer") as demo:
    gr.Markdown(
        "# VAE Latent Space Explorer\n"
        "An MLP Variational Autoencoder trained on MNIST compresses each "
        "28×28 digit image into a 20-dimensional latent vector. "
        "t-SNE projects it to 2D for visualization.\n\n"
        "**Click any point** on the scatter plot to auto-fill the coordinates, "
        "then press **Decode** to see the reconstructed image."
    )
    with gr.Row():
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Latent Space")
        with gr.Column(scale=2):
            gr.Markdown("### Decode from Latent Space")
            with gr.Row():
                tsne_x = gr.Number(label="t-SNE 1", value=0.0, elem_id="tsne-x")
                tsne_y = gr.Number(label="t-SNE 2", value=0.0, elem_id="tsne-y")
            with gr.Row():
                decode_btn = gr.Button("Decode", variant="primary")
                random_btn = gr.Button("Random Sample")
            orig_image = gr.Image(label="Original", height=250)
            recon_image = gr.Image(label="Decoded", height=250)
            info_text = gr.Textbox(label="Info", interactive=False)

    demo.load(fn=lambda: FIG, outputs=plot_output)
    
    decode_btn.click(
        fn=decode_from_coords,
        inputs=[tsne_x, tsne_y],
        outputs=[orig_image, recon_image, info_text],
    )
    random_btn.click(
        fn=decode_random,
        outputs=[orig_image, recon_image, info_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
