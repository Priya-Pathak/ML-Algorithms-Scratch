import torch
import numpy as np
import matplotlib.pyplot as plt

# Create 2 example RGB images (random for demo)
np.random.seed(42)
img1 = (np.random.rand(3, 5, 5) * 255).astype(np.uint8)
img2 = (np.random.rand(3, 5, 5) * 255).astype(np.uint8)

images = np.stack([img1, img2], axis=0)
images_torch = torch.tensor(images, dtype=torch.float32) / 255.0

# BatchNorm
def batch_norm_manual(x, eps=1e-5):
    N, C, H, W = x.shape
    x_flat = x.permute(1, 0, 2, 3).reshape(C, -1)
    mean = x_flat.mean(dim=1, keepdim=True)
    var = x_flat.var(dim=1, unbiased=False, keepdim=True)
    x_norm = (x_flat - mean) / torch.sqrt(var + eps)
    x_norm = x_norm.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return x_norm

normed = batch_norm_manual(images_torch)

def to_display(x):
    imgs = []
    for img in x:
        print(img.shape)
        img = img.clone()
        for c in range(3):
            minval = img[c].min()
            maxval = img[c].max()
            img[c] = (img[c] - minval) / (maxval - minval + 1e-8)
        imgs.append(np.transpose(img.numpy(), (1,2,0)))
    return imgs

orig_imgs = [np.transpose(images[0], (1,2,0)), np.transpose(images[1], (1,2,0))]
normed_imgs = to_display(normed)

# Show all four in one row
def show_images(imgs, titles):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

show_images(
    orig_imgs + normed_imgs,
    ['Original 1', 'Original 2', 'Normalized 1', 'Normalized 2']
)
