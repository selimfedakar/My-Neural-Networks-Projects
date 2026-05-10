import torch
import torch.nn.functional as F

# "Softmax comes in: I need to convert raw scores into valid probabilities."
# "Logits are flexible, but we need reconciliation with the real data."


# 1. Manual Softmax Implementation
logits = torch.tensor([2.0, 5.0, -1.0])  # Raw scores from the network

# Step A: Exponentiate — "counts = logits.exp() makes everything positive"
counts = logits.exp()

# Step B: Normalize — "probs = counts / counts.sum() guarantees a sum of 1.0"
probs = counts / counts.sum()


# 2. THE PROBLEM: Numerical Instability
# Very large logits cause exp(100) → inf, and inf/inf → NaN
# "The computer memory becomes infinity and our training crashes."
extreme_logits = torch.tensor([100.0, 100.0, 100.0])


# 3. THE SOLUTION: Cross Entropy (The "Gold Standard" Way)
# "In real-world Deep Learning, we stop manually calculating softmax, log, and mean."
# "We replace all with F.cross_entropy — numerically well-behaved under the hood."
target = torch.tensor([1])  # Correct character is at index 1 (score 5.0)
loss = F.cross_entropy(logits.unsqueeze(0), target)


if __name__ == "__main__":
    print(f"Manual Softmax Probs: {probs}")
    print("-" * 30)
    print(f"Cross Entropy Loss: {loss.item():.4f}")
    print("Successfully moved from 'manual math' to 'numerically stable' production code.")

# Note: this file and numerical_stability_test.py cover the same concept —
# stability_test goes deeper with experiments; this one is the condensed reference version.
