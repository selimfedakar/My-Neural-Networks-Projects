import torch

# "In bigram I were looking at 1 ch, in MLP I look at 3 ch because I define block-size=3."
# This is my context window: how many characters I look at to predict the next one.

#So, firstly;

# 1. Building the Dataset (Sliding Window)
# Example 'emma': [0,0,0] -> e(5), [0,0,5] -> m(13), [0,5,13] -> m(13)...
X = torch.tensor([
    [0, 0, 0],  # Context for 'e'
    [0, 0, 5],  # Context for 'm'
    [0, 5, 13]  # Context for 'm'
])
Y = torch.tensor([5, 13, 13]) # My targets

# 2. Lookup Table C (Word Embeddings)
# "The old solution was using one-hot encoding but now we use embeddings."
# I'm creating a 27x2 matrix to represent 27 characters in a 2-dimensional space.
C = torch.randn((27, 2)) # C is my library of vectors

# "emb = C[X] - Just use int tensors to pluck out rows from C."
emb = C[X]

print(f"Embedding Output Shape: {emb.shape}") # [3, 3, 2] -> 3 examples, 3 chars, 2 dim each

# 3. The Concatenation Problem
# "If we have [32, 3, 2], we can't multiply by a [6, 100] hidden layer. We must concatenate."
# I am squashing the 3 characters (each 2-dim) into a single 6-dimensional vector.
# Method from my notes: torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)
context_vector = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)

print(f"Concatenated Vector Shape: {context_vector.shape}") # [3, 6]

# 4. Moving to the Hidden Layer
W1 = torch.randn((6, 100)) # 6 inputs (3*2), 100 neurons
b1 = torch.randn(100)
h = (context_vector @ W1 + b1).tanh() # Activation as seen in my previous notes

print("-" * 30)
print(f"Hidden Layer Activation Shape: {h.shape}")
print("The network is now 'alive' and processing context!")