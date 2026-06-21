import torch

# Goal: check whether the learning rate is 'healthy' at any point during training.
# Insight from notes: the update-to-data ratio tells you if you're learning too fast or too slow.


def check_learning_health(lr, weight_tensor, grad_tensor):
    """
    Computes the ratio of update magnitude to weight magnitude.
    A ratio near 0.001 means the optimizer is taking appropriately sized steps.
    Too high: learning rate is blowing up the weights.
    Too low: learning rate is too small — weights barely move.
    """
    update_std = (lr * grad_tensor).std().item()
    data_std   = weight_tensor.std().item()
    ratio      = update_std / data_std

    print(f"Update-to-Data Ratio: {ratio:.6f}")

    # "The sweet spot is 0.001" — anything between 1e-3 +/- half an order of magnitude
    if 0.0005 < ratio < 0.005:
        print("Status: Gold Standard Learning")
    else:
        print("Status: Adjust your Learning Rate!")

    return ratio


if __name__ == "__main__":
    # Simulate a weight tensor and gradient tensor at some point during training
    W    = torch.randn(100, 100) * 0.1
    grad = torch.randn(100, 100) * 0.01

    print("--- Learning Rate Health Check ---")
    check_learning_health(lr=0.01,  weight_tensor=W, grad_tensor=grad)
    print()
    check_learning_health(lr=10.0,  weight_tensor=W, grad_tensor=grad)
    print()
    check_learning_health(lr=0.001, weight_tensor=W, grad_tensor=grad)
