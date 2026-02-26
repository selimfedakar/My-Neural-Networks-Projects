import torch
import matplotlib.pyplot as plt

# My goal: Keep the 'volume' of activations stable across layers.
# I realized that if weights are too big, neurons die; if too small, they vanish.

def check_activation_stats(fan_in, fan_out, use_kaiming=True):
    # 1. Setup: I'm using 1000 examples
    x = torch.randn(1000, fan_in)

    # 2. Initialization Strategy
    if use_kaiming:
        # "I normalize my weights by the square root of the fan-in"
        # Formula: 1 / sqrt(fan_in)
        w = torch.randn(fan_in, fan_out) / (fan_in ** 0.5)
    else:
        # Standard Gaussian (The 'Uncalibrated' way)
        w = torch.randn(fan_in, fan_out)

    # 3. Forward Pass through Tanh
    # "The soul of the neural network is the non-linearity"
    h = torch.tanh(x @ w)

    print(f"--- Strategy: {'Kaiming' if use_kaiming else 'Standard'} ---")
    print(f"Mean: {h.mean().item():.4f}")
    print(f"Std Dev (Volume): {h.std().item():.4f}")

    # "If std is too low, the network is 'thin'; if too high, neurons are 'dead'"
    return h


# --- THE COMPARISON ---
fan_in, fan_out = 784, 512

# Scenario A: Signal grows/shrinks uncontrollably
h_bad = check_activation_stats(fan_in, fan_out, use_kaiming=False)

# Scenario B: "The signal reaches all the way back to the first layer"
h_good = check_activation_stats(fan_in, fan_out, use_kaiming=True)

print("-" * 30)
print("I've successfully calibrated the network volume using the Square Root of Fan-in.")