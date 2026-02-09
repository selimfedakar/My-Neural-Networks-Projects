import torch

# I am starting with a much simpler building block: Character-level modeling.
# As I noted: "Language modeling is a program that has learned the statistical patterns of a language."

# For this demo, I'll use a tiny sample of names to build my counts matrix N.
words = ["selim", "la", "dc", "ai", "neural"]
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}  # itos: integer to string mapping for decoding

# 1. Counting Bigrams
# I am creating a 27x27 matrix to store the frequency of character pairs.
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        N[ix1, ix2] += 1

# 2. Normalization
# "Counting bigrams is a learning process."
# I calculate probabilities by dividing counts by the total for each row.
P = (N + 1).float()  # Adding 1 for model smoothing (preventing 0 probability)
P /= P.sum(1, keepdim=True)

# 3. The Sampling Loop
# "This code is the 'pay off' for everything I have done so far."
g = torch.Generator().manual_seed(2147483647)  # Reproducibility as noted

print("Generating new names based on bigram probabilities:")
for i in range(5):
    out = []
    ix = 0  # Always start with the special '.' token
    while True:
        # I pull the probability distribution for the current character
        p = P[ix]

        # The "Magic Step": Sampling from the distribution like a weighted 27-sided die.
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        if ix == 0:  # Break when we sample the end token '.'
            break
        out.append(itos[ix])

    print(f"Generated Name {i + 1}: {''.join(out)}")

# In my notes, I realized that for large datasets, we use 'batches'.
# "When your dataset is like a million examples, we pick out a random subset."