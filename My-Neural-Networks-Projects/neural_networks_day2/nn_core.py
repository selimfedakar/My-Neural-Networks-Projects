import random
# Import my Value class from Day 1 or include it here
# Based on my notes: "the grad attribute is the derivative"
from Day_01_Autograd_Basics.micrograd_lite import Value

class Module:
    """ Base class for all Neural Network components """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    """ A single neuron implementing w*x + b with tanh activation """
    def __init__(self, nin):
        # Weights initialized randomly: "initial weights are random" (Note p.2)
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # raw_output = sum(wi * xi) + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # Activation function: "squashes into [-1, 1]" (Note p.2)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    """ A collection of neurons forming a layer """
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
    """ Multi-Layer Perceptron: "composite function chained together" (Note p.1) """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# --- Verification of the 5-step training loop (Note p.2) ---
x = [2.0, 3.0, -1.0] # Input data
n = MLP(3, [4, 4, 1]) # 3 inputs, two hidden layers of 4, 1 output

# 1. Predict (Forward Pass)
ypred = n(x)

# 2. Check Error (Calculating Loss)
# Let's say target is 1.0
target = Value(1.0)
loss = (ypred + (target * -1)) * (ypred + (target * -1)) # Squared error

# 3. Find Gradients (Backpropagation)
# "Triggering backprop to find gradients of parameters" (Note p.2)
n.zero_grad()
loss.backward()

# 4. Update Weights (Nudging in the opposite direction of gradient)
learning_rate = 0.01
for p in n.parameters():
    p.data += -learning_rate * p.grad # Step 4 from your notes

print(f"Loss after 1 step: {loss.data:.4f}")

#Predict -> Check Error -> Find Gradients -> Update Weights - loop complete for now