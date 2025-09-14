import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate example images
np.random.seed(42)
img1 = (np.random.rand(3, 32, 32) * 255).astype(np.uint8)
img2 = (np.random.rand(3, 32, 32) * 255).astype(np.uint8)

images = np.stack([img1, img2], axis=0)                 # (2, 3, 32, 32)
images_torch = torch.tensor(images, dtype=torch.float32) / 255.0

def layer_norm_manual(x, eps=1e-5):
    # x shape: (N, C, H, W)
    N, C, H, W = x.shape
    x_norm = torch.zeros_like(x)
    for n in range(N):
        for h in range(H):
            for w in range(W):
                pixel = x[n, :, h, w]                   # Shape (C,)
                mean = pixel.mean()
                var = pixel.var(unbiased=False)
                x_norm[n, :, h, w] = (pixel - mean) / torch.sqrt(var + eps)
    return x_norm

normed = layer_norm_manual(images_torch)

def to_display(x):
    imgs = []
    for img in x:
        img = img.clone()
        for c in range(3):
            minval = img[c].min()
            maxval = img[c].max()
            img[c] = (img[c] - minval) / (maxval - minval + 1e-8)
        imgs.append(np.transpose(img.numpy(), (1,2,0)))
    return imgs

def to_diff_display(orig, normed):
    diffs = []
    for orig_img, norm_img in zip(orig, normed):
        diff = torch.abs(orig_img - norm_img)
        # min-max scale per channel for visualization
        for c in range(3):
            minval = diff[c].min()
            maxval = diff[c].max()
            diff[c] = (diff[c] - minval) / (maxval - minval + 1e-8)
        diffs.append(np.transpose(diff.numpy(), (1,2,0)))
    return diffs

orig_imgs = [np.transpose(images[0], (1,2,0)), np.transpose(images[1], (1,2,0))]
normed_imgs = to_display(normed)
diff_imgs = to_diff_display(images_torch, normed)

def show_images(imgs, titles):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

show_images(
    orig_imgs + normed_imgs + diff_imgs,
    ['Original 1', 'Original 2', 'LayerNorm 1', 'LayerNorm 2', 'Diff 1', 'Diff 2']
)
