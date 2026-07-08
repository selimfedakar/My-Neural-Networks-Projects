import math


class Value:
    """
    Represents a scalar value and its gradient for automatic differentiation.
    Based on handwritten study notes on Micrograd and Neural Networks.

    Why does this exist? Because before using PyTorch's autograd, I needed to
    understand what's actually happening under the hood — every gradient,
    every backward call, built from scratch.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)  # Connective tissue of the computation graph
        self._op = _op
        self.label = label

    def __repr__(self):
        # Useful during debugging — lets you inspect any node at a glance
        return f"Value(label={self.label!r}, data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Addition distributes the gradient equally to both inputs
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        # Handles cases like: 1 + Value(...) — Python falls back to this
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Each input's gradient = the other input's value × upstream gradient
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        # Handles cases like: 3 * Value(...) — same reason as __radd__
        return self * other

    def tanh(self):
        """
        Squashes values into (-1, 1) for non-linearity in neuron layers.
        Using math.tanh directly avoids computing exp(2x) twice —
        same result, cleaner arithmetic.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivative of tanh: (1 - tanh²(x)) — a classic result worth memorizing
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Triggers backpropagation through the topological sort of the graph.
        The magic happens here: gradients flow backwards through every node
        that contributed to the final output.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient of the output node — dL/dL = 1 by definition
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    # --- Verification with handwritten notes ---
    # Initializing variables from study notes: (a * b + c) → tanh
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    L = d.tanh()
    L.label = 'L'

    L.backward()

    print("--- Neural Network Day 1 Results ---")
    print(f"Final Output (L) data: {L.data:.4f}")
    print(f"Gradient dL/da: {a.grad:.4f}")
    print(f"Gradient dL/db: {b.grad:.4f}")
    print(f"Gradient dL/dc: {c.grad:.4f}")
