import torch
import torch.nn.functional as F

# "This is the important fundamental concept in Deep Learning for me."
# Raw integers don't work because characters are distinct identities, not quantities.

# 1. The Problem: Raw Integers
# If I have 'a' (1) and 'm' (13), multiplying them by weights makes 'm' look 13x larger.
# In reality, 'm' isn't "more" than 'a'; it's just different.

# 2. The Solution: One-Hot Encoding
# We convert an integer into a vector where only one position is '1' (Active).
x = torch.tensor([13]) # Let's say we have the character 'm' at index 13
x_encoded = F.one_hot(x, num_classes=27).float()

print(f"Original Index: {x.item()}")
print(f"One-Hot Vector Shape: {x_encoded.shape}")
print(f"One-Hot Vector: {x_encoded}")

# 3. Neural Network Compatibility
# "A neural net is a mathematical equation: y = x * w + b"
# By using a (1, 27) vector, we can multiply it by a (27, 27) weight matrix.
# This "plucks out" the specific row corresponding to our character.

W = torch.randn((27, 27)) # 27 neurons working in parallel
logits = x_encoded @ W     # Matrix multiplication for identity-based prediction

print("-" * 30)
print(f"Logits (Raw Scores) Shape: {logits.shape}")
print("We have now moved the counting model into the 'Algorithm Space'.")