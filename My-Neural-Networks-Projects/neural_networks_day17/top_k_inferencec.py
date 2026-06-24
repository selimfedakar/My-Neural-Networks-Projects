import torch
import torch.nn.functional as F

# Goal: implement a filter that keeps only the top-K most likely predictions.
# Instead of sampling from the full distribution, we silence the unlikely candidates.


def sample_top_k(logits, k=5):
    # 1. Find the values and indices of the top K logits
    v, i = torch.topk(logits, k)

    # 2. Build a mask of '-inf' for everything below the top K
    # "If I set them to -inf, softmax will turn them into 0 probability."
    out = torch.full_like(logits, float('-inf'))

    # 3. Fill back only the top K values — the rest remain silenced
    out.scatter_(1, i, v)

    # 4. Sample from the filtered distribution
    probs    = F.softmax(out, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1).item()

    return next_idx


if __name__ == "__main__":
    # Dummy logit vector for a vocab of 27 characters
    dummy_logits = torch.randn(1, 27)

    # "Filtering out the noise and keeping only the top 5 candidates."
    filtered_idx = sample_top_k(dummy_logits, k=5)

    print(f"Top-K Sampling selected index: {filtered_idx}")
    print("Successfully added the 'Noise Filter' to the RNN inference engine!")
