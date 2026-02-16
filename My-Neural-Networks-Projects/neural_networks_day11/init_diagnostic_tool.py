import torch


# My goal: Visualize why 'scale' (0.01) is the magic number to avoid "Initialization Mess."
# I need to ensure my neurons are "equally unsure" at the start.

def analyze_my_initialization(scale_factor):
    # 1. Setup: I am using 1000 examples and 500 neurons to see the statistical behavior.
    x = torch.randn(1000, 10)
    W = torch.randn(10, 500) * scale_factor  # I scale weights as documented in my notes.
    b = torch.randn(500) * 0.01

    # 2. Pre-activation (h_preact)
    # This is the raw score before I apply the tanh function.
    h_preact = x @ W + b

    # 3. Activation (h)
    # Applying Tanh. I noted: if h_preact is > 2 or < -2, tanh starts to "flat" out.
    h = torch.tanh(h_preact)

    # 4. Diagnostic: Counting 'Dead' Neurons
    # I consider a neuron 'dead' if its value is too close to -1 or 1.
    # In these regions, the gradient is 0, and my 'reasoning' stops.
    dead_percentage = (h.abs() > 0.99).float().mean() * 100

    print(f"--- My Analysis with Scale: {scale_factor} ---")
    print(f"Mean Activation: {h.mean().item():.4f}")
    print(f"Std Deviation: {h.std().item():.4f}")
    print(f"Dead Neurons: {dead_percentage:.2f}%")
    print("-" * 30)


# --- THE COMPARISON ---

# Scenario A: The 'Messy' Initialization
# I documented this as "Fake Confidence" which creates the "Hockey Stick" loss.
analyze_my_initialization(scale_factor=1.0)

# Scenario B: The 'Gold Standard' Initialization
# I use this to keep my neurons in the active learning region.
analyze_my_initialization(scale_factor=0.01)

# I remind myself: I can't set weights to 0 to avoid Symmetry Breaking.