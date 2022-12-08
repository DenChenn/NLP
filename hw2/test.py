import torch

# Create a tensor with random values
t = torch.randn(1, 256, 1)

# Print the shape of the tensor
print(t)

# Start with a tensor of shape (1, 256)
t = torch.randn(1, 256)

# Use squeeze to remove the singleton dimensions
t = torch.squeeze(t)

# Use unsqueeze to add a new dimension of size 1 at the specified index
t = torch.unsqueeze(t, dim=2)

# The tensor now has shape (1, 256, 1)
print(t)

