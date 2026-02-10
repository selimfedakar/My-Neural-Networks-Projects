import torch

# "Language modeling is a program that has learned the statistical patterns of a language."
# My goal for today: Mastering Broadcasting to avoid slow 'for' loops.

# 1. Setup the N Matrix (Counting Bigrams)
# Let's assume we have our 27x27 matrix as documented in my notes.
N = torch.randint(0, 100, (27, 27)) # Simulating counts with random numbers for now

# 2. The "Magic" of Broadcasting
# As I noted: "PyTorch does that virtually so the fast way is Broadcasting."

# Step A: Calculate Row Sums
# "N.sum(dim=1, keepdim=True) returns a (27, 1) column vector."
# "Without keepdim=True, it returns a (27,) vector which causes broadcasting issues."
row_sums = N.sum(dim=1, keepdim=True)

# Step B: Element-wise Division
# "PyTorch sees (27, 27) / (27, 1) and performs the division all at once."
# "This is incredibly fast because it runs in C, not Python."
P = N / row_sums

# 3. Verification of NLL (The Evaluation Point)
# "How to calculate the likelihood of the training data?"
# I need to ensure the sum of each row is exactly 1.0 (Probability Distribution).
print(f"Check: Sum of first row probabilities = {P[0].sum().item():.4f}")

# 4. Moving toward the Loss Function
# As I documented: "The next step is to use this score to calculate loss."
# "We want probabilities to be high, meaning the 'badness' (loss) should be low."

print("-" * 30)
print(f"P Matrix Shape: {P.shape}")
print(f"Broadcasting successfully created a {P.shape[0]}x{P.shape[1]} distribution.")