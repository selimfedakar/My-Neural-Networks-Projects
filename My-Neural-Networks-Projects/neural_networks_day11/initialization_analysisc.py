import torch

# "So in initialization, the loss is extremely high... creating fake confidence."
# Goal: achieve logits closer to 0 at the start so the model is "equally unsure."


# 1. THE SYMMETRY BREAKING PROBLEM
# "If I set W to 0, every single neuron becomes identical."
W_bad = torch.zeros((27, 100))
# Every neuron produces the same output → same gradient → learns the same thing.
# We need randomness to break symmetry so neurons specialise independently.


# 2. THE "DEAD TANH" RISK
# "If weights are initialized poorly, inputs to tanh might be very large."
# "In these regions, tanh is flat and the gradient becomes 0."
x        = torch.randn((1, 10))
W_large  = torch.randn((10, 100)) * 50   # Poor initialization — extreme scale
h_preact = x @ W_large
h        = torch.tanh(h_preact)
# If h is exactly ±1.0, gradient is 0 — the neuron is "dead" and stops learning.


# 3. THE "HOCKEY STICK" PROBLEM
# "If weights are too big, I get a 'hockey stick' loss curve."
# I waste the first few hundred iterations just waiting for weights to shrink
# before any real learning can begin.


# 4. THE GOLD STANDARD FIX: Small Initial Weights
W_good = torch.randn((27, 100)) * 0.01  # Small scale → avoids fake confidence & dead tanh
b_good = torch.zeros(100)               # Biases at 0 → no initial shift into flat tanh regions


if __name__ == "__main__":
    print(f"Dead neurons with W_large (abs > 0.99): {(h.abs() > 0.99).sum().item()}")
    print("-" * 30)
    print("Initialization Strategy:")
    print("  Weights: small random numbers — break symmetry, avoid Dead Tanh.")
    print("  Biases:  set to 0 — avoid shifting tanh into flat regions at step 0.")
