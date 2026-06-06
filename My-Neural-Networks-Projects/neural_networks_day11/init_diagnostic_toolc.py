import torch

# Goal: visualize why scale (0.01) is the magic number to avoid "Initialization Mess."
# Neurons should be "equally unsure" at the start — not confidently wrong.


def analyze_my_initialization(scale_factor):
    # 1. Setup: 1000 examples, 500 neurons — large enough to see statistical behavior
    x      = torch.randn(1000, 10)
    W      = torch.randn(10, 500) * scale_factor  # Scale as documented in notes
    b      = torch.zeros(500)                      # Biases start at 0 — no initial shift

    # 2. Pre-activation (h_preact)
    # Raw score before tanh — this is what initialization directly controls
    h_preact = x @ W + b

    # 3. Activation (h)
    # If h_preact is > 2 or < -2, tanh starts to "flatten out" and gradients vanish
    h = torch.tanh(h_preact)

    # 4. Diagnostic: Counting 'Dead' Neurons
    # A neuron is 'dead' when tanh saturates — gradient ≈ 0, no more learning
    dead_percentage = (h.abs() > 0.99).float().mean() * 100

    print(f"--- Analysis with Scale: {scale_factor} ---")
    print(f"Mean Activation:  {h.mean().item():.4f}")
    print(f"Std Deviation:    {h.std().item():.4f}")
    print(f"Dead Neurons:     {dead_percentage:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    # --- THE COMPARISON ---

    # Scenario A: The 'Messy' Initialization
    # "Fake Confidence" — large weights → tanh saturates → "Hockey Stick" loss curve
    analyze_my_initialization(scale_factor=1.0)

    # Scenario B: The 'Gold Standard' Initialization
    # Small weights keep neurons in the active learning region from the start
    analyze_my_initialization(scale_factor=0.01)

    # Reminder: weights cannot be set to 0 — that would cause symmetry breaking
    # (all neurons become identical and learn the same thing forever)
