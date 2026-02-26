import torch

# I understood that backpropagation is like a courier carrying a message from the loss.
# Gradient of variable = local gradient * gradient from above (Chain Rule).

# 1. Forward Pass
x = torch.tensor([-2.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# I am calculating the intermediate steps
z = x * w + b
h = torch.tanh(z) # My activation function

# 2. THE MANUAL BACKPROP (The Courier's Journey)
# Let's say the gradient from above (dLoss/dh) is 1.0
grad_from_above = 1.0

# Local gradient of tanh is (1 - tanh^2(z))
local_grad_tanh = 1 - h.data**2
grad_z = local_grad_tanh * grad_from_above # Message reaching 'z'

# Moving to weights and inputs
# dZ/dw = x, dZ/dx = w, dZ/db = 1
grad_w = x.data * grad_z
grad_x = w.data * grad_z
grad_b = 1.0 * grad_z

print(f"Manual Gradient for W: {grad_w.item():.4f}")

# 3. VERIFICATION
h.backward()
print(f"PyTorch Gradient for W: {w.grad.item():.4f}")
print("I've successfully simulated the courier reaching the weights!")