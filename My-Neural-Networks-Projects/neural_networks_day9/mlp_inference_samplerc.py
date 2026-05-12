import torch
import torch.nn.functional as F

# "I am implementing the 'Emma' logic from my notes: context = context[1:] + [ix]"
# Goal: use the MLP to generate new names by sliding the 3-character context window.

# 1. Model Setup (random weights — architecture demo, not a trained model)
block_size = 3  # The 'Context Window' defined in Day 9
C  = torch.randn((27, 2))
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
W2 = torch.randn((100, 27))
b2 = torch.randn(27)

g = torch.Generator().manual_seed(2147483647)


if __name__ == "__main__":
    # 2. THE SAMPLING LOOP: The Sliding Window in Action
    for _ in range(5):  # Generate 5 names
        out = []
        # "Imagine processing 'emma': context starts empty — [0, 0, 0]"
        context = [0] * block_size

        while True:
            # Step A: Forward pass — 'view' flattens [1, 3, 2] into [1, 6]
            emb    = C[torch.tensor([context])]       # [1, 3, 2]
            h      = torch.tanh(emb.view(1, -1) @ W1 + b1)  # [1, 100]
            logits = h @ W2 + b2                      # [1, 27]
            probs  = F.softmax(logits, dim=1)

            # Step B: Sample the next character
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()

            # Step C: The "Magic Logic" from page 7
            # "adds the new character to the end, drops the oldest one"
            context = context[1:] + [ix]

            if ix == 0:  # Stop when the '.' end-token is sampled
                break

            # Simplified itos: assumes vocab is a-z at indices 1-26
            out.append(chr(ord('a') + ix - 1))

        print(f"Generated Name: {''.join(out)}")

    print("-" * 30)
    print("Successfully moved from static context to a dynamic sampling loop!")
