import torch

# Starting with a much simpler building block: character-level modeling.
# "Language modeling is a program that has learned the statistical patterns of a language."

# Tiny sample dataset — in practice this would be thousands of names
words = ["selim", "la", "dc", "ai", "neural"]

# Build vocabulary dynamically — '.' is the special start/end token at index 0
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}  # itos: integer-to-string for decoding

vocab_size = len(stoi)  # Dynamic — no hardcoding, works for any character set


# 1. Counting Bigrams
# A vocab_size x vocab_size matrix stores how often each character pair appears
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        N[ix1, ix2] += 1


# 2. Normalization
# "Counting bigrams is a learning process."
# +1 smoothing prevents zero probability for unseen pairs (Laplace smoothing)
P = (N + 1).float()
P /= P.sum(1, keepdim=True)  # Each row sums to 1.0 — a proper probability distribution


if __name__ == "__main__":
    # 3. The Sampling Loop
    # "This code is the 'pay off' for everything I have done so far."
    g = torch.Generator().manual_seed(2147483647)  # Fixed seed for reproducibility

    print("Generating new names based on bigram probabilities:")
    for i in range(5):
        out = []
        ix = 0  # Always start from the '.' token
        while True:
            p = P[ix]

            # The "Magic Step": sampling from the distribution like a
            # weighted die with as many sides as our vocabulary
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

            if ix == 0:  # End token '.' sampled — name is complete
                break
            out.append(itos[ix])

        print(f"Generated Name {i + 1}: {''.join(out)}")

    # Note: for large datasets we use batches —
    # "When your dataset is like a million examples, we pick out a random subset."
