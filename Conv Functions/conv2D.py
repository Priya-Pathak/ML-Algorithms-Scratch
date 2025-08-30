import numpy as np

def conv2d(input_image, kernel, stride=1, padding=0):
    """
    Basic 2D convolution (single channel, no batch).
    Args:
        input_image : 2D numpy array (H x W)
        kernel      : 2D numpy array (kH x kW)
        stride      : int, stride of convolution
        padding     : int, zero-padding around input
    Returns:
        output      : 2D numpy array (output shape)
    """
    # Add padding
    if padding > 0:
        input_padded = np.pad(
            input_image,
            ((padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input_image

    H, W = input_padded.shape
    kH, kW = kernel.shape

    # Output size calculation
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    output = np.zeros((out_H, out_W))
    
    # Convolution loop
    for i in range(out_H):
        for j in range(out_W):
            region = input_padded[
                i*stride: i*stride+kH,
                j*stride: j*stride+kW
            ]
            output[i, j] = np.sum(region * kernel)
    return output

inp = np.array([[1,2,3,0],[0,1,2,3],[3,0,1,2],[2,3,0,1]])
ker = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
result = conv2d(inp, ker, stride=1, padding=1)
print('Input:')
print(inp)
print('Kernel')
print(ker)
print('Output')
print(result)
