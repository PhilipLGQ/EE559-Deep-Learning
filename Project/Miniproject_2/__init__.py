import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set autograd globally off
# torch.set_grad_enabled(False)
