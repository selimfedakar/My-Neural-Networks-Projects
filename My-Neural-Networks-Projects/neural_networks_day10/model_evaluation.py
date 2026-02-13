import torch

# "I am taking original X and y and splitting them into 3 buckets."
# tr  = training set (80%) -> To calculate gradients and update weights.
# dev = development/validation set (10%) -> To tune hyperparameters.
# te  = test set (10%) -> To evaluate final performance.

# 1. Simulating a Split (80/10/10)
# Let's assume I have 230k examples as documented in my notes.
total_examples = 230000
n1 = int(0.8 * total_examples)
n2 = int(0.9 * total_examples)

# tr = X[:n1], dev = X[n1:n2], te = X[n2:]
print(f"Training set: {n1} examples")
print(f"Validation set: {n2 - n1} examples")
print(f"Test set: {total_examples - n2} examples")

# 2. Performance Review
# "Old bigram model loss: 2.45 | Current MLP loss: 2.45"
# "This actually indicates UNDERFITTING."
# "Underfitting means the model is too small to capture complex patterns."

# 3. Parameter Counting
# "sum(p.nelement() for p in parameters)"
# For an MLP with 3.5k parameters vs 230k examples, the model is likely too small.

# 4. Strategic Suggestion from my notes:
# "To fix underfitting, we can increase: hidden layer size, embedding dim, or context window."

print("-" * 30)
print("Evaluation Complete: Loss is stagnant at 2.45.")
print("Moving to increase Hidden Layer size for better 'yetakleme' (capture).")