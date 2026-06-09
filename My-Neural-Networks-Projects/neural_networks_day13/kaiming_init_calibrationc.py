import torch
import matplotlib.pyplot as plt

# Goal: keep the 'volume' of activations stable across layers.
# If weights are too big, neurons die; if too small, signals vanish.


def check_activation_stats(fan_in, fan_out, use_kaiming=True):
    # 1. Setup: 1000 examples to observe statistical behavior at scale
    x = torch.randn(1000, fan_in)

    # 2. Initialization Strategy
    if use_kaiming:
        # "Normalize weights by the square root of fan-in" — keeps variance stable
        # Formula: 1 / sqrt(fan_in)  (Xavier/Glorot, works well with tanh)
        w = torch.randn(fan_in, fan_out) / (fan_in ** 0.5)
    else:
        # Standard Gaussian — the 'Uncalibrated' way
        w = torch.randn(fan_in, fan_out)

    # 3. Forward Pass through Tanh
    # "The soul of the neural network is the non-linearity"
    h = torch.tanh(x @ w)

    print(f"--- Strategy: {'Kaiming' if use_kaiming else 'Standard'} ---")
    print(f"Mean:               {h.mean().item():.4f}")
    print(f"Std Dev (Volume):   {h.std().item():.4f}")
    # "If std is too low, the network is 'thin'; if too high, neurons are 'dead'"

    return h


if __name__ == "__main__":
    fan_in, fan_out = 784, 512

    # Scenario A: Signal grows/shrinks uncontrollably
    h_bad  = check_activation_stats(fan_in, fan_out, use_kaiming=False)

    # Scenario B: "The signal reaches all the way back to the first layer"
    h_good = check_activation_stats(fan_in, fan_out, use_kaiming=True)

    print("-" * 30)
    print("Successfully calibrated network volume using the Square Root of Fan-in.")

    # Histogram comparison — visualizes the 'volume' difference between strategies
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(h_bad.flatten().detach().numpy(),  bins=50, color='salmon')
    axes[0].set_title("Standard Init — Activation Distribution")
    axes[1].hist(h_good.flatten().detach().numpy(), bins=50, color='steelblue')
    axes[1].set_title("Kaiming Init — Activation Distribution")
    plt.tight_layout()
    plt.show()
