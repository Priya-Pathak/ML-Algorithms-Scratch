import numpy as np

def pointwise_conv2d(input_tensor, weights):
    """
    Pointwise (1x1) convolution for multi-channel 2D inputs.
    Args:
        input_tensor (np.ndarray): Input of shape (H, W, C_in)
        weights      (np.ndarray): Weight of shape (C_in, C_out)
    Returns:
        output      (np.ndarray): Output of shape (H, W, C_out)
    """
    H, W, C_in = input_tensor.shape
    C_out = weights.shape[1]
    # Flatten spatial dimensions; do matrix multiply; reshape to output
    output = np.zeros((H, W, C_out))
    for i in range(H):
        for j in range(W):
            # For each spatial location, multiply C_in vector by weights (C_in x C_out)
            output[i, j, :] = np.dot(input_tensor[i, j, :], weights)
    return output

# Example usage:
inp = np.random.rand(5, 5, 3)            # (H, W, C_in)
weights = np.random.rand(3, 4)           # (C_in, C_out)
out = pointwise_conv2d(inp, weights)
print(out.shape)  # Should be (5, 5, 4)
