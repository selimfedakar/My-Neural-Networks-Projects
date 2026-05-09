import torch

# "Language modeling is a program that has learned the statistical patterns of a language."
# Goal for today: mastering broadcasting to avoid slow Python for-loops.

VOCAB_SIZE = 27  # 26 letters + 1 special token '.'


# 1. Setup the N Matrix (Counting Bigrams)
# Simulating counts with random numbers — in practice this comes from real data
N = torch.randint(0, 100, (VOCAB_SIZE, VOCAB_SIZE))


# 2. The "Magic" of Broadcasting
# "PyTorch does that virtually so the fast way is Broadcasting."

# Step A: Calculate Row Sums
# "N.sum(dim=1, keepdim=True) returns a (27, 1) column vector."
# "Without keepdim=True, it returns a (27,) vector which causes broadcasting issues."
row_sums = N.sum(dim=1, keepdim=True)

# Step B: Element-wise Division
# "PyTorch sees (27, 27) / (27, 1) and performs the division all at once."
# "This is incredibly fast because it runs in C, not Python."
P = N.float() / row_sums


if __name__ == "__main__":
    # 3. Verification of NLL (The Evaluation Point)
    # Each row must sum to 1.0 — the sanity check for any probability distribution
    print(f"Check: Sum of first row probabilities = {P[0].sum().item():.4f}")

    # 4. Moving Toward the Loss Function
    # "The next step is to use this score to calculate loss."
    # "We want probabilities to be high, meaning the 'badness' (loss) should be low."
    print("-" * 30)
    print(f"P Matrix Shape: {P.shape}")
    print(f"Broadcasting successfully created a {P.shape[0]}x{P.shape[1]} distribution.")
