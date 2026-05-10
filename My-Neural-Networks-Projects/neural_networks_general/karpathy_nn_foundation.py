# =============================================================================
# Neural Network Foundations — Reference Guide
# Based on Andrej Karpathy's micrograd lecture and notebook
# Original source: https://github.com/karpathy/micrograd
# All code structure and pedagogy credit: Andrej Karpathy (MIT License)
# Collected here as a personal reference — the cleanest mental model for
# understanding backpropagation from first principles.
# =============================================================================

import math


# -----------------------------------------------------------------------------
# PART 1: What is a derivative?
# The core intuition before any neural network code.
# -----------------------------------------------------------------------------

def derivative_intuition():
    """
    f(x) = 3x^2 - 4x + 5
    df/dx = limit of [f(x+h) - f(x)] / h  as h -> 0
    This is the slope of f at a point x — how much f responds to a nudge in x.
    """
    def f(x):
        return 3*x**2 - 4*x + 5

    x  = 3.0
    h  = 0.0001
    df = (f(x + h) - f(x)) / h   # numerical gradient
    print(f"f({x})   = {f(x)}")
    print(f"f'({x})  ≈ {df:.4f}")   # should be close to 6x - 4 = 14

    # Multiple inputs — partial derivatives
    a, b, c = 2.0, -3.0, 10.0
    d1 = a * b + c
    da = ((a + h) * b + c - d1) / h
    db = (a * (b + h) + c - d1) / h
    dc = (a * b + (c + h) - d1) / h
    print(f"\nd = a*b + c = {d1}")
    print(f"∂d/∂a = {da:.4f}  (should be b = {b})")
    print(f"∂d/∂b = {db:.4f}  (should be a = {a})")
    print(f"∂d/∂c = {dc:.4f}  (should be 1.0)")


# -----------------------------------------------------------------------------
# PART 2: The Value class — scalar autograd engine
# Each Value wraps a float and tracks the computation graph.
# _backward holds the local chain-rule rule for this operation.
# -----------------------------------------------------------------------------

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data  = data
        self.grad  = 0.0           # dL/d(self) — starts at zero
        self._backward = lambda: None
        self._prev = set(_children)
        self._op   = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- forward operations ---

    def __add__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        out    = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # gradient flows through addition unchanged (local grad = 1)
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # d(a*b)/da = b,  d(a*b)/db = a
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f'**{exponent}')
        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t   = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            # d(tanh)/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e   = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # convenience wrappers so normal Python arithmetic works
    def __neg__(self):         return self * -1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other**-1

    # --- backward pass ---

    def backward(self):
        """
        Topological sort of the graph → reverse order → call each node's
        _backward.  The seed gradient of the output node is 1.0.
        """
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# -----------------------------------------------------------------------------
# PART 3: Manual backprop through a single neuron (Karpathy's walkthrough)
# Shows exactly what .backward() computes, step by step.
# -----------------------------------------------------------------------------

def single_neuron_manual():
    # Inputs and weights
    x1, x2 = Value(2.0, label='x1'), Value(0.0, label='x2')
    w1, w2 = Value(-3.0, label='w1'), Value(1.0, label='w2')
    b       = Value(6.8813735870195432, label='b')

    # Forward pass:  o = tanh(x1*w1 + x2*w2 + b)
    x1w1 = x1 * w1;  x1w1.label = 'x1w1'
    x2w2 = x2 * w2;  x2w2.label = 'x2w2'
    xw   = x1w1 + x2w2;  xw.label = 'x1w1+x2w2'
    n    = xw + b;   n.label = 'n'
    o    = n.tanh(); o.label = 'o'

    # Backward pass: chain rule from output to inputs
    o.backward()

    print("After backward():")
    for v in [x1, w1, x2, w2, b, o]:
        print(f"  {v.label:8s}  data={v.data:8.4f}  grad={v.grad:8.4f}")

    # Key insight from Karpathy:
    # x1.grad = -1.5 means: increasing x1 by a tiny amount decreases the output
    # by 1.5x that amount.  That IS the gradient — the sensitivity of the loss.


# -----------------------------------------------------------------------------
# PART 4: Neural network modules built on Value
# Neuron → Layer → MLP  (Karpathy's exact architecture)
# -----------------------------------------------------------------------------

import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)} inputs)"


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({self.neurons})"


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP({self.layers})"


# -----------------------------------------------------------------------------
# PART 5: Training loop
# Loss → backward → nudge parameters → zero gradients → repeat
# -----------------------------------------------------------------------------

def training_demo():
    random.seed(42)
    model = MLP(3, [4, 4, 1])   # 3 inputs, two hidden layers of 4, 1 output

    # Tiny dataset
    xs = [
        [2.0,  3.0, -1.0],
        [3.0, -1.0,  0.5],
        [0.5,  1.0,  1.0],
        [1.0,  1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]   # target labels

    for step in range(20):
        # Forward
        ypred = [model(x) for x in xs]
        loss  = sum((yp - Value(yt))**2 for yp, yt in zip(ypred, ys))

        # Backward
        for p in model.parameters():
            p.grad = 0.0           # zero gradients before backward
        loss.backward()

        # Update (gradient descent)
        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 5 == 0:
            print(f"step {step:2d}  loss={loss.data:.6f}")

    print("\nPredictions vs targets:")
    for x, yt, yp in zip(xs, ys, [model(x) for x in xs]):
        print(f"  target={yt:5.1f}  pred={yp.data:6.4f}")


# -----------------------------------------------------------------------------
# PART 6: PyTorch equivalence — same neuron, same result
# Karpathy's sanity-check: micrograd and PyTorch must agree.
# -----------------------------------------------------------------------------

def pytorch_equivalence_note():
    """
    import torch
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double();  w2.requires_grad = True
    b  = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True

    o  = torch.tanh(x1*w1 + x2*w2 + b)
    o.backward()

    # x1.grad, w1.grad  should match Value.grad above
    """
    print("See docstring — PyTorch produces identical gradients to Value.backward().")
    print("This is the whole point: PyTorch IS micrograd, scaled to tensors.")


# -----------------------------------------------------------------------------
# Key lessons (Karpathy's summary)
# 1. Every operation has a simple local gradient rule.
# 2. Chain rule chains them all the way back to the inputs.
# 3. Topological sort ensures children are processed before parents (in reverse).
# 4. PyTorch does exactly this on tensors — same math, GPU accelerated.
# 5. The loss function is the single scalar we differentiate w.r.t. all weights.
# 6. Gradient descent: weights -= lr * grad, repeatedly.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Derivative intuition")
    print("=" * 60)
    derivative_intuition()

    print("\n" + "=" * 60)
    print("PART 2-3: Single neuron — manual backprop")
    print("=" * 60)
    single_neuron_manual()

    print("\n" + "=" * 60)
    print("PART 5: MLP training loop")
    print("=" * 60)
    training_demo()

    pytorch_equivalence_note()
