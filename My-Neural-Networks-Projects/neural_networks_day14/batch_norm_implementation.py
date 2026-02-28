import torch


# My goal: Force hidden states to be Gaussian and track running statistics.
# "I realized that moving data takes time, so I must handle these buffers carefully."

def simulate_batch_norm_layer(x, gamma, beta, bn_mean_running, bn_std_running, eps=1e-5):
    # 1. Calculate Batch Statistics
    # "Hidden activations now depend on every other example in the batch."
    batch_mean = x.mean(0, keepdim=True)  # mean across the batch
    batch_std = x.std(0, keepdim=True)  # std across the batch

    # 2. Force the Normalization
    # x_hat = (x - mean) / std
    x_hat = (x - batch_mean) / (batch_std + eps)

    # 3. Apply Gain and Bias (Gamma & Beta)
    # "This highlights the 'soul' of the neural network."
    out = gamma * x_hat + beta

    # 4. Update Running Statistics for Inference
    # "I keep 99.9% of what I knew and update only 0.1%."
    with torch.no_grad():
        bn_mean_running = 0.999 * bn_mean_running + 0.001 * batch_mean
        bn_std_running = 0.999 * bn_std_running + 0.001 * batch_std

    return out, bn_mean_running, bn_std_running


# --- THE TEST ---
# 32 examples, 100 neurons
x = torch.randn(32, 100) * 10 + 5  # High variance, shifted mean

# Parameters
gamma = torch.ones((1, 100))
beta = torch.zeros((1, 100))
running_mean = torch.zeros((1, 100))
running_std = torch.ones((1, 100))

# "I am mathematically forcing them to be roughly Gaussian."
out, running_mean, running_std = simulate_batch_norm_layer(x, gamma, beta, running_mean, running_std)

print(f"Original Mean: {x.mean().item():.4f}")
print(f"Normalized Mean: {out.mean().item():.4f} (Near 0!)")
print(f"Updated Running Mean (First 5): {running_mean[0, :5]}")
print("-" * 30)
print("I've successfully implemented the 'Unnatural Coupling' for training stability.")