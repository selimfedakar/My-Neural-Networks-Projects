import torch
import torch.nn.functional as F


# My Goal: Implement a filter that keeps only the top K most likely predictions.

def sample_top_k(logits, k=5):
    # 1. Identify the values and indices of the top K logits
    v, i = torch.topk(logits, k)

    # 2. Create a mask of '-infinity' for everything else
    # "I realized that if I set them to -inf, softmax will turn them into 0 probability."
    out = torch.full_like(logits, float('-inf'))

    # 3. Fill back only the top K values
    out.scatter_(1, i, v)

    # 4. Now perform Softmax and Sample
    probs = F.softmax(out, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1).item()

    return next_idx


# --- THE SIMULATION ---
# A dummy logit vector for a vocab of 27 characters
dummy_logits = torch.randn(1, 27)

# "I am filtering out the noise and keeping only the top 5 candidates."
filtered_idx = sample_top_k(dummy_logits, k=5)

print(f"Top-K Sampling selected index: {filtered_idx}")
print("I've successfully added the 'Noise Filter' to my RNN inference engine!")