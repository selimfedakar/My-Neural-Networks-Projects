import torch
import torch.nn.functional as F

# "I am implementing the 'Emma' logic from my notes: context = context[1:] + [ix]"
# My goal: Use the MLP to generate new names by sliding the 3-character window.

# 1. Model Setup (Assume I have trained C, W1, b1, W2, b2)
block_size = 3  # The 'Context Window' I defined in Day 9
C = torch.randn((27, 2))
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
W2 = torch.randn((100, 27))
b2 = torch.randn(27)

g = torch.Generator().manual_seed(2147483647)

# 2. THE SAMPLING LOOP: The Sliding Window in Action
for _ in range(5):  # Generate 5 names
    out = []
    # "Imagine processing emma, context starts empty: [0, 0, 0]"
    context = [0] * block_size

    while True:
        # Step A: Forward pass using the 'view' trick for efficiency
        emb = C[torch.tensor([context])]  # [1, 3, 2]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)  # [1, 100]
        logits = h @ W2 + b2  # [1, 27]
        probs = F.softmax(logits, dim=1)

        # Step B: Sample the next character
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()

        # Step C: The "Magic Logic" from page 7
        # "adds the new ch to the end, drop the oldest one"
        context = context[1:] + [ix]

        if ix == 0:  # Stop if we reach the '.' token
            break
        # itos lookup (simplified here)
        out.append(chr(ord('a') + ix - 1) if ix > 0 else '.')

    print(f"Generated Name: {''.join(out)}")

print("-" * 30)
print("I have successfully moved from static context to a dynamic sampling loop!")