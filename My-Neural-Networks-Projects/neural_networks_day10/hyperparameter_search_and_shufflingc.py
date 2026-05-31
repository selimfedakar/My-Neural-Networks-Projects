import torch

# "I realized that the reason we split data is because I cannot use all data for just training."
# Goal: use the 'dev' set to decide between two model sizes and fix underfitting.


# 1. SHUFFLING: The "Haphazard" Principle
# Ensuring buckets are not biased by the original order of the dataset
total_examples = 230000
g = torch.Generator().manual_seed(2147483647)
indices = torch.randperm(total_examples, generator=g)

# 80/10/10 split applied to the shuffled indices
n1 = int(0.8 * total_examples)
n2 = int(0.9 * total_examples)

train_indices = indices[:n1]
dev_indices   = indices[n1:n2]
test_indices  = indices[n2:]


# 2. HYPERPARAMETER SEARCH (The 'Dev' Bucket Logic)
# Two options to fix the underfitting diagnosed earlier:
#   Model A: Small (~3.5k parameters) — the one currently underfitting
#   Model B: Large (increased hidden layer size) — more capacity to capture patterns

def simulate_training_run(model_name):
    """Returns simulated dev-set loss — stand-in for a real training run."""
    results = {
        "Model_A": 2.45,  # Stagnant loss as noted
        "Model_B": 2.12,  # Improved loss due to better capacity
    }
    return results[model_name]


if __name__ == "__main__":
    # 3. THE DECISION POINT — using the dev set to make a principled choice
    loss_a = simulate_training_run("Model_A")
    loss_b = simulate_training_run("Model_B")

    print("--- Hyperparameter Tuning on Dev Set ---")
    print(f"Model A (Small) Dev Loss: {loss_a}")
    print(f"Model B (Large) Dev Loss: {loss_b}")

    if loss_b < loss_a:
        print("Decision: Proceeding with Model B — it effectively reduces underfitting.")
    else:
        print("Decision: Keep searching for better hyperparameters.")

    print("-" * 30)
    print("Note: Saving the 'Test Set' for the absolute final evaluation.")
