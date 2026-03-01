import torch
import torch.nn as nn


# My goal: Ensure the signal 'x' can bypass layers if they become a bottleneck.
# This is the 'Shortcut' or 'Skip Connection' that revolutionized deep networks.

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # "I'm using the standard pipeline: Linear -> BatchNorm -> Tanh"
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 1. The Standard Path (f(x))
        out = self.linear(x)
        out = self.bn(out)
        out = self.tanh(out)

        # 2. THE SKIP CONNECTION (x + f(x))
        # "If the linear layer starts 'killing' the gradient, x still carries the message."
        return x + out


# --- THE TEST ---
# 32 examples, 100-dimensional features
x = torch.randn(32, 100)
res_block = ResidualBlock(100)

# Forward pass
output = res_block(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print("-" * 30)
# "I verified that the output maintains the same dimension to allow for the addition."
print("I've successfully created a shortcut for my gradient courier!")