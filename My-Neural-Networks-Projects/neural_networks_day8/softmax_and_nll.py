import torch
import torch.nn.functional as F

# "Softmax comes in: I need to convert raw scores into valid probabilities."
# "Logits are flexible, but we need reconciliation with the real data."

# 1. Manual Softmax Implementation
logits = torch.tensor([2.0, 5.0, -1.0]) # Raw scores from my network

# Step A: Exponentiate to get positive 'counts'
# "counts = logits.exp() - this makes everything positive"
counts = logits.exp()

# Step B: Normalize to get probabilities
# "probs = counts / counts.sum() - guaranteed sum of 1.0"
probs = counts / counts.sum()

print(f"Manual Softmax Probs: {probs}")

# 2. THE PROBLEM: Numerical Instability
# If I have a very large score, exp(100) will be massive and crash the memory.
extreme_logits = torch.tensor([100.0, 100.0, 100.0])
# "The computer memory becomes infinity and our training crashes."

# 3. THE SOLUTION: Cross Entropy (The 'Gold Standard' Way)
# "In real-world Deep Learning, we stop manually calculating softmax, log, and mean."
# "We replace all with F.cross_entropy which is numerically well-behaved."

target = torch.tensor([1]) # Let's say the correct character is at index 1 ('5.0')
loss = F.cross_entropy(logits.unsqueeze(0), target)

print("-" * 30)
print(f"Cross Entropy Loss: {loss.item():.4f}")
print("I've successfully moved from 'manual math' to 'numerically stable' production code.")