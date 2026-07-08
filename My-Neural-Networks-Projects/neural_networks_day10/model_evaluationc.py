import torch

# "I am taking original X and y and splitting them into 3 buckets."
# tr  = training set   (80%) → to calculate gradients and update weights
# dev = validation set (10%) → to tune hyperparameters
# te  = test set       (10%) → to evaluate final performance (touch only once)


# 1. Simulating a Split (80/10/10)
# Assume 230k examples as documented in notes
total_examples = 230000
n1 = int(0.8 * total_examples)
n2 = int(0.9 * total_examples)

# Actual split would be: tr = X[:n1], dev = X[n1:n2], te = X[n2:]


# 2. Performance Review
# "Old bigram model loss: 2.45 | Current MLP loss: 2.45"
# "This actually indicates UNDERFITTING."
# Underfitting: the model is too small to capture the complexity of the data.

# 3. Parameter Counting
# Diagnostic: sum(p.nelement() for p in parameters)
# ~3.5k parameters vs 230k training examples → model lacks capacity

# 4. Strategic Fix (from notes):
# To fix underfitting, increase: hidden layer size, embedding dim, or context window.


if __name__ == "__main__":
    print(f"Training set:   {n1} examples")
    print(f"Validation set: {n2 - n1} examples")
    print(f"Test set:       {total_examples - n2} examples")
    print("-" * 30)
    print("Evaluation Complete: Loss is stagnant at 2.45.")
    print("Moving to increase Hidden Layer size for better pattern capture.")
