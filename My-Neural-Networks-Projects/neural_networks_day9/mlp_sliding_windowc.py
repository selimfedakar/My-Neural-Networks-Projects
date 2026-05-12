import torch

# "In bigram I was looking at 1 character; in MLP I look at 3 because I define block_size=3."
# This is the context window: how many characters I look at to predict the next one.


# 1. Building the Dataset (Sliding Window)
# Example 'emma': [0,0,0] -> e(5), [0,0,5] -> m(13), [0,5,13] -> m(13)...
X = torch.tensor([
    [0,  0,  0],   # Context for 'e'
    [0,  0,  5],   # Context for 'm'
    [0,  5, 13],   # Context for 'm'
])
Y = torch.tensor([5, 13, 13])  # Targets


# 2. Lookup Table C (Character Embeddings)
# "The old solution was using one-hot encoding but now we use embeddings."
# A 27x2 matrix represents 27 characters as 2-dimensional vectors.
C = torch.randn((27, 2))  # C is my library of vectors

# "emb = C[X] — just use integer tensors to pluck out rows from C."
emb = C[X]  # Shape: [3, 3, 2] — 3 examples, context of 3 chars, each 2-dimensional


# 3. The Concatenation Problem
# "If we have [3, 3, 2], we can't multiply by a [6, 100] hidden layer. We must concatenate."
# Three 2-dim character vectors are flattened into a single 6-dimensional context vector.
# Explicit form (from notes) — equivalent shortcut: emb.view(emb.shape[0], -1)
context_vector = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim=1)


# 4. Hidden Layer
W1 = torch.randn((6, 100))   # 6 inputs (block_size * embed_dim), 100 neurons
b1 = torch.randn(100)
h = (context_vector @ W1 + b1).tanh()


if __name__ == "__main__":
    print(f"Embedding Output Shape:    {emb.shape}")            # [3, 3, 2]
    print(f"Concatenated Vector Shape: {context_vector.shape}") # [3, 6]
    print("-" * 30)
    print(f"Hidden Layer Activation Shape: {h.shape}")          # [3, 100]
    print("The network is now 'alive' and processing context!")
