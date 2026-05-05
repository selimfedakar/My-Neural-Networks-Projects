import random
import sys
import os

# Day 1's Value class lives one level up — path adjusted for repo structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from micrograd_lite import Value


class Module:
    """Base class for all neural network components. Mirrors PyTorch's nn.Module pattern."""

    def zero_grad(self):
        # Reset all gradients before each backward pass — otherwise they accumulate
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """
    A single neuron: computes w·x + b, then squashes into [-1, 1] via tanh.
    "Initial weights are random" — Note p.2: avoids symmetry so neurons learn differently.
    """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __repr__(self):
        return f"Neuron(nin={len(self.w)})"

    def __call__(self, x):
        # raw = sum(wi * xi) + b — using self.b as start value folds bias in cleanly
        raw = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Activation: "squashes into [-1, 1]" — adds the non-linearity neurons need
        return raw.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A collection of neurons — each fires independently on the same input."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __repr__(self):
        return f"Layer({self.neurons})"

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # Unwrap single-neuron layers so the output isn't needlessly wrapped in a list
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """
    Multi-Layer Perceptron: "a composite function chained together" — Note p.1.
    Each layer feeds into the next, building up increasingly abstract representations.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __repr__(self):
        return f"MLP({self.layers})"

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    # --- Verification of the 5-step training loop (Note p.2) ---
    x = [2.0, 3.0, -1.0]        # Input
    n = MLP(3, [4, 4, 1])        # 3 inputs → two hidden layers of 4 → 1 output
    target = Value(1.0)

    # 1. Predict (Forward Pass)
    ypred = n(x)

    # 2. Check Error — squared error: (ypred - target)²
    # Note: Value doesn't have __sub__ yet, so we compute (ypred + (-1 * target))²
    diff = ypred + Value(-1.0) * target
    loss = diff * diff

    # 3. Find Gradients (Backpropagation)
    n.zero_grad()
    loss.backward()

    # 4. Update Weights — nudge each parameter opposite to its gradient
    learning_rate = 0.01
    for p in n.parameters():
        p.data -= learning_rate * p.grad

    # 5. Report
    print(f"Loss after 1 step: {loss.data:.4f}")
    # Predict → Check Error → Find Gradients → Update Weights — loop complete for now
