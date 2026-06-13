import torch

# Goal: force hidden states to be Gaussian and track running statistics.
# "I realized that moving data takes time, so I must handle these buffers carefully."


def simulate_batch_norm_layer(x, gamma, beta, bn_mean_running, bn_std_running, eps=1e-5):
    # 1. Calculate Batch Statistics
    # "Hidden activations now depend on every other example in the batch."
    batch_mean = x.mean(0, keepdim=True)
    # unbiased=False: batch norm divides by N, not N-1 (no Bessel correction)
    batch_std  = x.std(0,  keepdim=True, unbiased=False)

    # 2. Force the Normalization — x_hat = (x - mean) / std
    # eps prevents division by zero when std ≈ 0
    x_hat = (x - batch_mean) / (batch_std + eps)

    # 3. Apply Gain and Bias (Gamma & Beta)
    # Learnable scale and shift — "this highlights the 'soul' of the neural network"
    out = gamma * x_hat + beta

    # 4. Update Running Statistics for Inference
    # "I keep 99.9% of what I knew and update only 0.1%."
    # Running stats are not used during training — only at inference time.
    with torch.no_grad():
        bn_mean_running = 0.999 * bn_mean_running + 0.001 * batch_mean
        bn_std_running  = 0.999 * bn_std_running  + 0.001 * batch_std

    return out, bn_mean_running, bn_std_running


if __name__ == "__main__":
    # --- THE TEST ---
    # 32 examples, 100 neurons — high variance and shifted mean to stress-test normalization
    x = torch.randn(32, 100) * 10 + 5

    gamma        = torch.ones((1, 100))
    beta         = torch.zeros((1, 100))
    running_mean = torch.zeros((1, 100))
    running_std  = torch.ones((1, 100))

    # "I am mathematically forcing them to be roughly Gaussian."
    out, running_mean, running_std = simulate_batch_norm_layer(
        x, gamma, beta, running_mean, running_std
    )

    print(f"Original Mean:                    {x.mean().item():.4f}")
    print(f"Normalized Mean:                  {out.mean().item():.4f}  (Near 0!)")
    print(f"Updated Running Mean (First 5):   {running_mean[0, :5]}")
    print("-" * 30)
    print("Successfully implemented the 'Unnatural Coupling' for training stability.")
