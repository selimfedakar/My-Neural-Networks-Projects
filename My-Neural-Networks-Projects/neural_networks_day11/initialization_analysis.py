import torch
import torch.nn.functional as F

# "So in initialization, the loss is extremely high... creating fake confidence."
# My goal: Achieve logits closer to 0 at the start so the model is "equally unsure."

# 1. THE SYMMETRY BREAKING PROBLEM
# "If I set W to 0, every single neuron becomes identical."
W_bad = torch.zeros((27, 100))
# This is bad. We need randomness to "break symmetry" so neurons learn different things.

# 2. THE "DEAD TANH" RISK
# "If weights are initialized poorly, the inputs to tanh might be very large."
# "In these regions, the tanh is flat, and the gradient becomes 0."
x = torch.randn((1, 10))
W_large = torch.randn((10, 100)) * 50 # Poor initialization
h_preact = x @ W_large
h = torch.tanh(h_preact)

# If h is exactly 1.0 or -1.0, the gradient is 0. The neuron is "dead."
print(f"Number of dead neurons (abs val > 0.99): {(h.abs() > 0.99).sum().item()}")

# 3. THE "HOCKEY STICK" PROBLEM
# "If weights are too big, I get a 'hockey stick' loss curve."
# I waste the first few hundred iterations just waiting for weights to shrink.

# 4. THE GOLD STANDARD FIX: Small Initial Weights
W_good = torch.randn((27, 100)) * 0.01 # Scaling down to avoid fake confidence
b_good = torch.zeros(100) # Biases are usually fine at 0

print("-" * 30)
print("Initialization Strategy:")
print("- Weights: Small random numbers to break symmetry and avoid Dead Tanh.")
print("- Biases: Set to 0 to avoid shifting the tanh into flat regions.")