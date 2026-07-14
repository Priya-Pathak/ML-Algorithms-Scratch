# VAE Latent Space Explorer — Hugging Face Spaces

A Gradio app that visualizes the latent space of a Variational Autoencoder trained on MNIST.
Deploy it to [Hugging Face Spaces](https://huggingface.co/spaces) or run it locally.

## What It Does

1. **Encodes** all 10,000 MNIST test images into 20-dimensional latent vectors using the trained VAE encoder.
2. **Reduces** the 20D latent space to 2D with t-SNE for visualization.
3. **Plots** an interactive Plotly scatter colored by digit class.
4. **Click-to-decode** — click any point on the scatter to see the original image and its reconstruction from the VAE decoder.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio app (entry point for HF Spaces) |
| `requirements.txt` | pip dependencies for HF Spaces |
| `README.md` | This file |

The VAE model weights are downloaded automatically from the [VAE repo](https://github.com/Priya-Pathak/VAE) at first launch.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

The app will:
1. Download `best_model.pt` from GitHub (~3 MB, cached after first run).
2. Download MNIST test set if not present.
3. Run t-SNE on all 10K test encodings (~10 seconds).
4. Launch the Gradio server (typically at `http://127.0.0.1:7860`).

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) with **Gradio** as the SDK.
2. Upload `app.py` and `requirements.txt`.
3. The Space will auto-build and launch.

No GPU needed — the VAE is small (MLP) and all inference runs on CPU.

## Model Architecture

```
Encoder:  Linear(784, 500) -> ReLU -> mu_head / logvar_head
Latent:   20-dimensional (reparameterization trick)
Decoder:  Linear(20, 500) -> ReLU -> Linear(500, 784) -> Sigmoid
Loss:     BCE (reconstruction) + KL divergence
```

Trained for 200 epochs on MNIST with Adagrad (lr=0.01).
See the full training code in [VAE/VAE/vae_mnist.py](https://github.com/Priya-Pathak/VAE/blob/main/VAE/vae_mnist.py).
