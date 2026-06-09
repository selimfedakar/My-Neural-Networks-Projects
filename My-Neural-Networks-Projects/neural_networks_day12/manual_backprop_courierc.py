import torch

# Backpropagation is like a courier carrying a message from the loss.
# Gradient of variable = local gradient × gradient from above (Chain Rule).


# 1. Forward Pass
x = torch.tensor([-2.0], requires_grad=True)
w = torch.tensor([0.5],  requires_grad=True)
b = torch.tensor([1.0],  requires_grad=True)

z = x * w + b
h = torch.tanh(z)  # Activation function


# 2. THE MANUAL BACKPROP (The Courier's Journey)

# grad_from_above = 1.0 because dLoss/dLoss = 1 — the seed that starts the chain
grad_from_above = 1.0

# Local gradient of tanh: (1 - tanh²(z))
local_grad_tanh = 1 - h.data ** 2
grad_z = local_grad_tanh * grad_from_above  # Message reaching 'z'

# Chain rule applied to each input of z = x*w + b
# dZ/dw = x,  dZ/dx = w,  dZ/db = 1
grad_w = x.data * grad_z
grad_x = w.data * grad_z
grad_b = grad_z


# 3. VERIFICATION — does PyTorch agree?
h.backward()

if __name__ == "__main__":
    print(f"Manual Gradient for W:   {grad_w.item():.4f}")
    print(f"PyTorch Gradient for W:  {w.grad.item():.4f}")
    print("Successfully simulated the courier reaching the weights!")
