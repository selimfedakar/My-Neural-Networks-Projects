import torch
import torch.nn.functional as F

# "This is the important fundamental concept in Deep Learning for me."
# Raw integers don't work because characters are distinct identities, not quantities.

VOCAB_SIZE = 27  # 26 letters + 1 special token '.'


# 1. The Problem: Raw Integers
# If I have 'a' (1) and 'm' (13), multiplying them by weights makes 'm' look 13x larger.
# In reality, 'm' isn't "more" than 'a'; it's just different.

# 2. The Solution: One-Hot Encoding
# Convert an integer into a vector where only one position is '1' (Active).
x = torch.tensor([13])  # Character 'm' at index 13 in the 27-char vocabulary
x_encoded = F.one_hot(x, num_classes=VOCAB_SIZE).float()

# 3. Neural Network Compatibility
# "A neural net is a mathematical equation: y = x * w + b"
# A (1, 27) vector multiplied by a (27, 27) weight matrix "plucks out"
# the specific row corresponding to our character — identity-based lookup.
W = torch.randn((VOCAB_SIZE, VOCAB_SIZE))  # 27 neurons working in parallel
logits = x_encoded @ W                     # Matrix multiplication → raw scores


if __name__ == "__main__":
    print(f"Original Index: {x.item()}")
    print(f"One-Hot Vector Shape: {x_encoded.shape}")
    print(f"One-Hot Vector: {x_encoded}")
    print("-" * 30)
    print(f"Logits (Raw Scores) Shape: {logits.shape}")
    print("We have now moved the counting model into the 'Algorithm Space'.")
