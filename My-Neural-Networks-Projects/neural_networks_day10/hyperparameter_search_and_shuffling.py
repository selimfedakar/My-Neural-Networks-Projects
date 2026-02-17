import torch

# "I realized that the reason we split data is because I cannot use all data for just training."
# My goal: Use the 'dev' set to decide between two model sizes (fixing underfitting).

# 1. SHUFFLING: The "Haphazard" Principle
# I need to ensure my buckets are not biased by the order of the dataset.
total_examples = 230000
# I am creating a random permutation of indices
g = torch.Generator().manual_seed(2147483647)
indices = torch.randperm(total_examples, generator=g)

# Using my previous 80/10/10 split logic on these random indices
n1 = int(0.8 * total_examples)
n2 = int(0.9 * total_examples)

train_indices = indices[:n1]
dev_indices = indices[n1:n2]
test_indices = indices[n2:]

# 2. HYPERPARAMETER SEARCH (The 'Dev' Bucket Logic)
# I have two options to fix the underfitting I diagnosed earlier:
# Model A: Small (3.5k parameters) - The one currently underfitting.
# Model B: Large (Increased Hidden Layer Size as suggested in my notes).

def simulate_training_run(model_name, dataset_indices):
    # This simulates the final loss on the dev set after training.
    # Model B has more 'capacity' to capture patterns.
    results = {
        "Model_A": 2.45, # Stagnant loss as noted
        "Model_B": 2.12  # Improved loss due to better 'yetakleme' (capture).
    }
    return results[model_name]

# 3. THE DECISION POINT
# I am using the DEV set to make a professional choice.
loss_a = simulate_training_run("Model_A", dev_indices)
loss_b = simulate_training_run("Model_B", dev_indices)

print(f"--- Hyperparameter Tuning on Dev Set ---")
print(f"Model A (Small) Dev Loss: {loss_a}")
print(f"Model B (Large) Dev Loss: {loss_b}")

if loss_b < loss_a:
    print("Decision: I am proceeding with Model B because it effectively reduces underfitting.")
else:
    print("Decision: I will keep searching for better hyperparameters.")

print("-" * 30)
print("Note: I am saving the 'Test Set' for the absolute final evaluation.")