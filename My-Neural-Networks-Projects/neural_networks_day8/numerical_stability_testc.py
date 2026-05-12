import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# DISCOVERY NOTES: Numerical Stability in Deep Learning
# Today, I proved to myself why manual Softmax can be dangerous
# and why F.cross_entropy is the "Gold Standard" for production.
# ---------------------------------------------------------


if __name__ == "__main__":

    print("--- EXPERIMENT 1: When Life is Easy (Small Numbers) ---")
    # At first, everything seemed fine. Manual softmax works perfectly with small values.
    logits = torch.tensor([2.0, 5.0, -1.0])
    counts = logits.exp()
    probs = counts / counts.sum()
    print(f"My Calculated Probs: {probs}")
    print("Result: The math works perfectly here.\n")


    print("--- EXPERIMENT 2: The System Crash (The 'inf' Problem) ---")
    # But what if my model produces massive values by mistake? (e.g., 100.0)
    # exp(100.0) exceeds float32 range — the result is 'inf', and inf/inf = NaN.
    extreme_logits = torch.tensor([100.0, 100.0, 100.0])
    extreme_counts = extreme_logits.exp()
    manual_fail = extreme_counts / extreme_counts.sum()
    print(f"Extreme Logits: {extreme_logits}")
    print(f"Manual Calculation Result: {manual_fail}")
    print("Note: Oh no! The result is 'NaN' (Not a Number). Training would crash here.\n")


    print("--- EXPERIMENT 3: I Found the Trick (Log-Sum-Exp) ---")
    # Subtracting the maximum value from all logits keeps the ratio identical!
    # Mathematically: Softmax(x) == Softmax(x - max(x))
    # This way, the largest value becomes 0.0, and exp(0.0) is always 1.0. Safe!
    max_logit = extreme_logits.max()
    stable_logits = extreme_logits - max_logit  # Now it's [0.0, 0.0, 0.0]
    stable_counts = stable_logits.exp()
    stable_probs = stable_counts / stable_counts.sum()
    print(f"My Normalized Logits: {stable_logits}")
    print(f"My Stabilized Probs: {stable_probs} -> (System Saved!)\n")


    print("--- EXPERIMENT 4: PyTorch's Wisdom ---")
    # PyTorch already applies this trick inside F.cross_entropy —
    # manual softmax + log + mean collapsed into one numerically stable call.
    target = torch.tensor([0])  # Correct class is at index 0
    loss = F.cross_entropy(extreme_logits.unsqueeze(0), target)
    print(f"PyTorch's Stable Loss: {loss.item():.4f}")
    print("Closing Note: I'm moving from manual math to the 'well-behaved' F.cross_entropy.")
    print("Stability test passed successfully.")
