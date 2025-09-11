# A typical training loop workflow
# The standard sequence of operations in a PyTorch training loop for a single mini-batch is: 
# optimizer.zero_grad(): 
# This clears any previously computed gradients from the model's parameters, ensuring a clean slate for the new mini-batch.
# output = model(input): 
# The forward pass computes the model's predictions for the current mini-batch.
# loss = criterion(output, target): 
# The loss function calculates the difference between the predictions and the actual targets.
# loss.backward(): 
# The backward pass computes the gradient of the loss with respect to every model parameter that requires_grad=True. These new gradients are stored in the .grad attribute of the parameter tensors.
# optimizer.step(): 
# The optimizer uses the gradients currently stored in the parameters to update the weights of the model according to its specific algorithm (e.g., Adam, SGD). 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SingleKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SingleKernelConv2d, self).__init__()
        # Initialize a single kernel as a learnable parameter
        # The shape of the kernel is (out_channels, in_channels, kernel_height, kernel_width)
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) # Optional: Add a bias
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Perform 2D convolution using the custom kernel
        return F.conv2d(x, self.kernel, self.bias, self.stride, self.padding)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Example: 1 input channel (grayscale), 1 output channel, 3x3 kernel
        self.conv1 = SingleKernelConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


# Create a dummy input image (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 28, 28) # Example: 1 grayscale image, 28x28

# Instantiate the model
model = SimpleCNN()

# Define a loss function and optimizer
criterion = nn.MSELoss() # Or any suitable loss for your task
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (simplified for demonstration)
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(dummy_input)
    # Assuming a target output for demonstration
    target = torch.randn_like(output)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nTraining finished.")
print("Learned kernel weights:\n", model.conv1.kernel.data)
print("Learned bias: \n", model.conv1.bias.data)