import numpy as np
import matplotlib.pyplot as plt

def dynamic_kernel(image):
    # Create a 3x3 kernel from image statistics (mean, std)
    mean = np.mean(image)
    std = np.std(image)
    kernel = np.array([
        [mean, std, mean],
        [std,  1,  std ],
        [mean, std, mean]
    ])
    kernel /= np.sum(np.abs(kernel))
    return kernel

def dynamic_conv2d(image):
    H, W = image.shape
    kernel = dynamic_kernel(image)
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    img_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)
    for i in range(H):
        for j in range(W):
            region = img_padded[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)
    return output, kernel

# Create random image
img = (np.random.rand(20, 20) * 255).astype(np.float32)

# Apply dynamic convolution
out, kern = dynamic_conv2d(img)

# Display original, convolution result, and kernel
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(out, cmap='gray')
axes[1].set_title("After Dynamic Conv")
axes[1].axis('off')

im = axes[2].imshow(kern, cmap='viridis')
axes[2].set_title("Dynamic Kernel")
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
