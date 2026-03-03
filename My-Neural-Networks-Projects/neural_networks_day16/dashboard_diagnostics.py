import torch


# My goal: Check if the learning rate is 'healthy'
def check_learning_health(lr, weight_tensor, grad_tensor):
    # Calculate the step size (update)
    update_std = (lr * grad_tensor).std().item()
    data_std = weight_tensor.std().item()

    # Calculate the ratio
    ratio = update_std / data_std

    print(f"Update-to-Data Ratio: {ratio:.6f}")

    # "The sweet spot is 0.001"
    if 0.0005 < ratio < 0.005:
        print("✅ Status: Gold Standard Learning")
    else:
        print("⚠️ Status: Adjust your Learning Rate!")